/*
 * ds-nlm-denoise – GStreamer / DeepStream 7.1 plugin header
 * Target: NVIDIA Jetson Orin  (sm_87)
 *
 * Non-Local Means denoiser with Lucas-Kanade Optical Flow
 */

#ifndef __GST_DS_NLM_DENOISE_H__
#define __GST_DS_NLM_DENOISE_H__

#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "gst-nvquery.h"
#include "gstnvdsmeta.h"

#include "customer_functions.h"

#define PACKAGE         "ds_nlm_denoise"
#define VERSION         "1.0"
#define LICENSE         "Proprietary"
#define DESCRIPTION     "DeepStream 7.1 NLM + Optical-Flow denoiser for Jetson Orin"
#define BINARY_PACKAGE  "NVIDIA DeepStream NLM Denoise"
#define URL             "https://github.com/your-org/ds-nlm-denoise"

G_BEGIN_DECLS

typedef struct _GstDsNlmDenoise      GstDsNlmDenoise;
typedef struct _GstDsNlmDenoiseClass GstDsNlmDenoiseClass;

#define GST_TYPE_DS_NLM_DENOISE \
    (gst_ds_nlm_denoise_get_type())
#define GST_DS_NLM_DENOISE(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_DS_NLM_DENOISE, GstDsNlmDenoise))
#define GST_DS_NLM_DENOISE_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_DS_NLM_DENOISE, GstDsNlmDenoiseClass))
#define GST_IS_DS_NLM_DENOISE(obj) \
    (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_DS_NLM_DENOISE))
#define GST_IS_DS_NLM_DENOISE_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_DS_NLM_DENOISE))
#define GST_DS_NLM_DENOISE_CAST(obj) \
    ((GstDsNlmDenoise *)(obj))

/** Maximum batch size supported by the plugin */
#define NLM_MAX_BATCH_SIZE 1024

struct _GstDsNlmDenoise
{
    GstBaseTransform base_trans;

    /* NLM processing hooks */
    CustomerFunction nlm_functions;

    /* CUDA context */
    CUcontext cuda_ctx;

    /* Element unique ID */
    guint unique_id;

    /* Frame counter */
    guint64 frame_num;

    /* CUDA stream for async kernel launches */
    cudaStream_t cuda_stream;

    /* Intermediate surface for format conversions */
    NvBufSurface *inter_buf;

    /* Video info parsed from caps */
    GstVideoInfo video_info;

    /* Resolution at which frames are processed */
    gint processing_width;
    gint processing_height;

    /* iGPU flag */
    guint is_integrated;

    /* Max batch size */
    guint max_batch_size;

    /* Target GPU id */
    guint gpu_id;

    /* Enable / disable the denoiser at runtime */
    gboolean enable_denoise;

    /* NvBufSurfTransform config */
    NvBufSurfTransformConfigParams transform_config_params;
};

struct _GstDsNlmDenoiseClass
{
    GstBaseTransformClass parent_class;
};

GType gst_ds_nlm_denoise_get_type(void);

G_END_DECLS
#endif /* __GST_DS_NLM_DENOISE_H__ */
