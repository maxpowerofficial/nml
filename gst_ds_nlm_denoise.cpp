/*
 * ds-nlm-denoise – GStreamer plugin implementation
 * DeepStream 7.1 / Jetson Orin
 *
 * This file is intentionally kept thin – all heavy processing lives in
 * nlm_denoise.cu.  The plugin just wires up GStreamer boilerplate,
 * maps NvBufSurface → EGLImage, calls the CUDA denoiser, and unmaps.
 */

#include <string.h>
#include <string>
#include <iostream>
#include "gst_ds_nlm_denoise.h"
#include <sys/time.h>

GST_DEBUG_CATEGORY_STATIC(gst_ds_nlm_denoise_debug);
#define GST_CAT_DEFAULT gst_ds_nlm_denoise_debug

static GQuark _dsmeta_quark = 0;

/* ------------------------------------------------------------------ */
/* Property IDs                                                         */
/* ------------------------------------------------------------------ */
enum {
    PROP_0,
    PROP_UNIQUE_ID,
    PROP_PROCESSING_WIDTH,
    PROP_PROCESSING_HEIGHT,
    PROP_ENABLE_DENOISE,
    PROP_BATCH_SIZE,
    PROP_GPU_DEVICE_ID
};

/* ------------------------------------------------------------------ */
/* Default property values                                              */
/* ------------------------------------------------------------------ */
#define DEFAULT_UNIQUE_ID          20
#define DEFAULT_PROCESSING_WIDTH   1920
#define DEFAULT_PROCESSING_HEIGHT  1080
#define DEFAULT_ENABLE_DENOISE     TRUE
#define DEFAULT_GPU_ID             0
#define DEFAULT_BATCH_SIZE         1

/* ------------------------------------------------------------------ */
/* Helper macros                                                        */
/* ------------------------------------------------------------------ */
#define CHECK_NVDS_MEMORY_AND_GPUID(obj, surf) \
({ int _err = 0; \
   do { \
     if ((surf->memType == NVBUF_MEM_DEFAULT || \
          surf->memType == NVBUF_MEM_CUDA_DEVICE) && \
         surf->gpuId != obj->gpu_id) { \
       GST_ELEMENT_ERROR(obj, RESOURCE, FAILED, \
           ("Surface gpu-id (%d) != plugin gpu-id (%d)", \
            surf->gpuId, obj->gpu_id), (NULL)); \
       _err = 1; \
     } \
   } while(0); \
   _err; \
})

#define CHECK_CUDA_STATUS(status, msg) \
  do { \
    if ((status) != cudaSuccess) { \
      g_print("CUDA error: %s in %s:%d (%s)\n", \
              msg, __FILE__, __LINE__, cudaGetErrorName(status)); \
      goto error; \
    } \
  } while(0)

#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"

/* ------------------------------------------------------------------ */
/* Pad templates                                                        */
/* ------------------------------------------------------------------ */
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE(
    "sink", GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE_WITH_FEATURES(
        GST_CAPS_FEATURE_MEMORY_NVMM, "{ NV12, RGBA, I420 }")));

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE(
    "src", GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE_WITH_FEATURES(
        GST_CAPS_FEATURE_MEMORY_NVMM, "{ NV12, RGBA, I420 }")));

/* ------------------------------------------------------------------ */
/* GObject boilerplate                                                  */
/* ------------------------------------------------------------------ */
#define gst_ds_nlm_denoise_parent_class parent_class
G_DEFINE_TYPE(GstDsNlmDenoise, gst_ds_nlm_denoise, GST_TYPE_BASE_TRANSFORM);

static void gst_ds_nlm_denoise_set_property(GObject*, guint, const GValue*, GParamSpec*);
static void gst_ds_nlm_denoise_get_property(GObject*, guint, GValue*, GParamSpec*);
static gboolean gst_ds_nlm_denoise_set_caps(GstBaseTransform*, GstCaps*, GstCaps*);
static gboolean gst_ds_nlm_denoise_start(GstBaseTransform*);
static gboolean gst_ds_nlm_denoise_stop(GstBaseTransform*);
static GstFlowReturn gst_ds_nlm_denoise_transform_ip(GstBaseTransform*, GstBuffer*);

/* ------------------------------------------------------------------ */
/* Class init                                                           */
/* ------------------------------------------------------------------ */
static void
gst_ds_nlm_denoise_class_init(GstDsNlmDenoiseClass *klass)
{
    GObjectClass          *go  = (GObjectClass*)klass;
    GstElementClass       *ge  = (GstElementClass*)klass;
    GstBaseTransformClass *bt  = (GstBaseTransformClass*)klass;

    g_setenv("DS_NEW_BUFAPI", "1", TRUE);

    go->set_property = GST_DEBUG_FUNCPTR(gst_ds_nlm_denoise_set_property);
    go->get_property = GST_DEBUG_FUNCPTR(gst_ds_nlm_denoise_get_property);

    bt->set_caps     = GST_DEBUG_FUNCPTR(gst_ds_nlm_denoise_set_caps);
    bt->start        = GST_DEBUG_FUNCPTR(gst_ds_nlm_denoise_start);
    bt->stop         = GST_DEBUG_FUNCPTR(gst_ds_nlm_denoise_stop);
    bt->transform_ip = GST_DEBUG_FUNCPTR(gst_ds_nlm_denoise_transform_ip);

    g_object_class_install_property(go, PROP_UNIQUE_ID,
        g_param_spec_uint("unique-id", "Unique ID",
            "Unique element ID (for metadata tagging)",
            0, G_MAXUINT, DEFAULT_UNIQUE_ID,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(go, PROP_PROCESSING_WIDTH,
        g_param_spec_int("processing-width", "Processing Width",
            "Input buffer width passed to the denoiser",
            1, G_MAXINT, DEFAULT_PROCESSING_WIDTH,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(go, PROP_PROCESSING_HEIGHT,
        g_param_spec_int("processing-height", "Processing Height",
            "Input buffer height passed to the denoiser",
            1, G_MAXINT, DEFAULT_PROCESSING_HEIGHT,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(go, PROP_ENABLE_DENOISE,
        g_param_spec_boolean("enable-denoise", "Enable Denoise",
            "Enable or disable the NLM denoiser at runtime",
            DEFAULT_ENABLE_DENOISE,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(go, PROP_GPU_DEVICE_ID,
        g_param_spec_uint("gpu-id", "GPU Device ID",
            "GPU device to run CUDA kernels on",
            0, G_MAXUINT, DEFAULT_GPU_ID,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
                          GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property(go, PROP_BATCH_SIZE,
        g_param_spec_uint("batch-size", "Batch Size",
            "Maximum batch size for processing",
            1, NLM_MAX_BATCH_SIZE, DEFAULT_BATCH_SIZE,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
                          GST_PARAM_MUTABLE_READY)));

    gst_element_class_add_pad_template(ge,
        gst_static_pad_template_get(&src_template));
    gst_element_class_add_pad_template(ge,
        gst_static_pad_template_get(&sink_template));

    gst_element_class_set_details_simple(ge,
        "ds_nlm_denoise",
        "NLM + Optical-Flow Denoise Plugin",
        DESCRIPTION,
        "NVIDIA Corporation <support@nvidia.com>");
}

/* ------------------------------------------------------------------ */
/* Instance init                                                        */
/* ------------------------------------------------------------------ */
static void
gst_ds_nlm_denoise_init(GstDsNlmDenoise *self)
{
    self->unique_id          = DEFAULT_UNIQUE_ID;
    self->processing_width   = DEFAULT_PROCESSING_WIDTH;
    self->processing_height  = DEFAULT_PROCESSING_HEIGHT;
    self->enable_denoise     = DEFAULT_ENABLE_DENOISE;
    self->gpu_id             = DEFAULT_GPU_ID;
    self->max_batch_size     = DEFAULT_BATCH_SIZE;
    self->inter_buf          = NULL;
    self->cuda_ctx           = NULL;
    self->cuda_stream        = NULL;
    self->frame_num          = 0;

    if (!_dsmeta_quark)
        _dsmeta_quark = g_quark_from_static_string(NVDS_META_STRING);
}

/* ------------------------------------------------------------------ */
/* Properties                                                           */
/* ------------------------------------------------------------------ */
static void
gst_ds_nlm_denoise_set_property(GObject *obj, guint id,
                                  const GValue *val, GParamSpec *ps)
{
    GstDsNlmDenoise *self = GST_DS_NLM_DENOISE(obj);
    switch (id) {
        case PROP_UNIQUE_ID:         self->unique_id         = g_value_get_uint(val);    break;
        case PROP_PROCESSING_WIDTH:  self->processing_width  = g_value_get_int(val);     break;
        case PROP_PROCESSING_HEIGHT: self->processing_height = g_value_get_int(val);     break;
        case PROP_ENABLE_DENOISE:    self->enable_denoise    = g_value_get_boolean(val); break;
        case PROP_GPU_DEVICE_ID:     self->gpu_id            = g_value_get_uint(val);    break;
        case PROP_BATCH_SIZE:        self->max_batch_size    = g_value_get_uint(val);    break;
        default: G_OBJECT_WARN_INVALID_PROPERTY_ID(obj, id, ps); break;
    }
}

static void
gst_ds_nlm_denoise_get_property(GObject *obj, guint id,
                                  GValue *val, GParamSpec *ps)
{
    GstDsNlmDenoise *self = GST_DS_NLM_DENOISE(obj);
    switch (id) {
        case PROP_UNIQUE_ID:         g_value_set_uint(val,    self->unique_id);         break;
        case PROP_PROCESSING_WIDTH:  g_value_set_int(val,     self->processing_width);  break;
        case PROP_PROCESSING_HEIGHT: g_value_set_int(val,     self->processing_height); break;
        case PROP_ENABLE_DENOISE:    g_value_set_boolean(val, self->enable_denoise);    break;
        case PROP_GPU_DEVICE_ID:     g_value_set_uint(val,    self->gpu_id);            break;
        case PROP_BATCH_SIZE:        g_value_set_uint(val,    self->max_batch_size);    break;
        default: G_OBJECT_WARN_INVALID_PROPERTY_ID(obj, id, ps); break;
    }
}

/* ------------------------------------------------------------------ */
/* Caps negotiation                                                     */
/* ------------------------------------------------------------------ */
static gboolean
gst_ds_nlm_denoise_set_caps(GstBaseTransform *bt,
                              GstCaps *incaps, GstCaps * /*outcaps*/)
{
    GstDsNlmDenoise *self = GST_DS_NLM_DENOISE(bt);
    GstVideoInfo vi;

    if (!gst_video_info_from_caps(&vi, incaps)) {
        GST_ELEMENT_ERROR(self, STREAM, FAILED,
            ("Failed to parse input caps"), (NULL));
        return FALSE;
    }
    self->video_info = vi;
    GST_DEBUG_OBJECT(self, "Caps set: %dx%d @ %d/%d fps",
                     vi.width, vi.height, vi.fps_n, vi.fps_d);
    return TRUE;
}

/* ------------------------------------------------------------------ */
/* Start / Stop                                                         */
/* ------------------------------------------------------------------ */
static gboolean
gst_ds_nlm_denoise_start(GstBaseTransform *bt)
{
    GstDsNlmDenoise *self = GST_DS_NLM_DENOISE(bt);

    if (cudaStreamCreate(&self->cuda_stream) != cudaSuccess) {
        GST_ELEMENT_ERROR(self, RESOURCE, FAILED,
            ("Failed to create CUDA stream"), (NULL));
        return FALSE;
    }

    /* Initialise the NLM denoiser from nlm_denoise.cu */
    init(&self->nlm_functions);

    GST_DEBUG_OBJECT(self, "NLM denoise plugin started");
    return TRUE;
}

static gboolean
gst_ds_nlm_denoise_stop(GstBaseTransform *bt)
{
    GstDsNlmDenoise *self = GST_DS_NLM_DENOISE(bt);

    if (self->inter_buf) {
        NvBufSurfaceDestroy(self->inter_buf);
        self->inter_buf = NULL;
    }
    if (self->cuda_stream) {
        cudaStreamDestroy(self->cuda_stream);
        self->cuda_stream = NULL;
    }

    GST_DEBUG_OBJECT(self, "NLM denoise plugin stopped");
    return TRUE;
}

/* ------------------------------------------------------------------ */
/* Per-buffer processing                                                */
/* ------------------------------------------------------------------ */
static GstFlowReturn
gst_ds_nlm_denoise_transform_ip(GstBaseTransform *bt, GstBuffer *inbuf)
{
    GstDsNlmDenoise *self = GST_DS_NLM_DENOISE(bt);
    GstMapInfo map_info;
    GstFlowReturn ret = GST_FLOW_ERROR;
    NvBufSurface *surface = NULL;
    NvDsBatchMeta *batch_meta = NULL;

    if (!self->enable_denoise)
        return GST_FLOW_OK;

    if (!gst_buffer_map(inbuf, &map_info, GST_MAP_READ)) {
        GST_ELEMENT_ERROR(self, STREAM, FAILED,
            ("Cannot map input buffer"), (NULL));
        return ret;
    }

    surface = (NvBufSurface*)map_info.data;

    if (CHECK_NVDS_MEMORY_AND_GPUID(self, surface))
        goto error;

    self->is_integrated =
        (surface->memType == NVBUF_MEM_SURFACE_ARRAY) ? 1 : 0;

    batch_meta = gst_buffer_get_nvds_batch_meta(inbuf);
    if (!batch_meta)
        GST_DEBUG_OBJECT(self, "No batch meta on this buffer");

    self->frame_num++;

    /* Map EGLImage for each surface in the batch */
    for (guint i = 0; i < surface->batchSize; i++) {
        if (!surface->surfaceList[i].mappedAddr.eglImage)
            NvBufSurfaceMapEglImage(surface, i);
    }

    /* Run NLM denoiser on every frame in the batch */
    if (batch_meta) {
        for (NvDsFrameMetaList *l = batch_meta->frame_meta_list;
             l != NULL; l = l->next) {

            NvDsFrameMeta *fm = (NvDsFrameMeta*)l->data;
            guint idx = fm->batch_id;
            if (idx >= surface->batchSize) continue;

            EGLImageKHR img =
                (EGLImageKHR)surface->surfaceList[idx].mappedAddr.eglImage;
            if (img && self->nlm_functions.fGPUProcess)
                self->nlm_functions.fGPUProcess(img, NULL);
        }
    } else {
        /* Fallback: process all surfaces even without metadata */
        for (guint i = 0; i < surface->batchSize; i++) {
            EGLImageKHR img =
                (EGLImageKHR)surface->surfaceList[i].mappedAddr.eglImage;
            if (img && self->nlm_functions.fGPUProcess)
                self->nlm_functions.fGPUProcess(img, NULL);
        }
    }

    /* Unmap EGLImage */
    for (guint i = 0; i < surface->batchSize; i++) {
        if (surface->surfaceList[i].mappedAddr.eglImage)
            NvBufSurfaceUnMapEglImage(surface, i);
    }

    ret = GST_FLOW_OK;

error:
    nvds_set_output_system_timestamp(inbuf, GST_ELEMENT_NAME(self));
    gst_buffer_unmap(inbuf, &map_info);
    return ret;
}

/* ------------------------------------------------------------------ */
/* Plugin registration                                                  */
/* ------------------------------------------------------------------ */
static gboolean
ds_nlm_denoise_plugin_init(GstPlugin *plugin)
{
    GST_DEBUG_CATEGORY_INIT(gst_ds_nlm_denoise_debug,
                            "ds_nlm_denoise", 0,
                            "DeepStream NLM Denoise plugin");

    return gst_element_register(plugin, "ds_nlm_denoise",
                                GST_RANK_PRIMARY,
                                GST_TYPE_DS_NLM_DENOISE);
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR, GST_VERSION_MINOR,
    nvdsgst_ds_nlm_denoise,
    DESCRIPTION,
    ds_nlm_denoise_plugin_init,
    VERSION, LICENSE, BINARY_PACKAGE, URL)
