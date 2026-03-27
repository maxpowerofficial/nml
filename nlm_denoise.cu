/*
 * ds-nlm-denoise — CUDA Non-Local Means denoiser with Lucas-Kanade
 * Optical Flow for DeepStream 7.1 / Jetson Orin
 *
 * Algorithm overview
 * ------------------
 *  1. Lucas-Kanade optical flow (Lucas & Kanade, 1981)  – same lightweight
 *     gradient-based approach used in the original project, but exposed
 *     here as a first-class step so that NLM patch search is
 *     motion-compensated across frames.
 *  2. Non-Local Means denoising (Buades et al., 2005)
 *     Every output pixel y(i) is a weighted average of pixels inside a
 *     search window.  The weight between two candidate patches p and q is:
 *
 *         w(p,q) = exp( -||I(p) - I(q)||^2 / (h^2) )
 *
 *     where h is the filter strength (larger h → stronger smoothing).
 *     Motion-compensated NLM: the reference patch centre for each output
 *     pixel is shifted by the optical-flow vector so that moving edges
 *     are preserved rather than blurred.
 *  3. Chroma smoothing: simple IIR low-pass on U/V channels (fast,
 *     effective for colour noise).
 *
 * Memory layout
 * -------------
 *  NV12 surface (iGPU / Jetson): Y plane full-res, UV plane half-height
 *  interleaved.  All processing stays on-device.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaEGL.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "customer_functions.h"

/* ------------------------------------------------------------------ */
/* Tunable constants stored in CUDA constant memory                    */
/* ------------------------------------------------------------------ */
__constant__ float c_NLM_H;           /* filter strength (e.g. 12.0) */
__constant__ int   c_PATCH_RADIUS;    /* half-size of comparison patch */
__constant__ int   c_SEARCH_RADIUS;   /* half-size of search window    */
__constant__ float c_MOTION_THRESH;   /* px/frame – switch static/motion path */
__constant__ float c_CHROMA_ALPHA;    /* IIR coefficient for chroma    */
__constant__ float c_TEMPORAL_BLEND;  /* blend weight for prev-frame NLM result */

/* ------------------------------------------------------------------ */
/* Per-instance state                                                  */
/* ------------------------------------------------------------------ */
struct NlmState {
    uint8_t *Y_prev_dev;    /* previous luma plane (pitched)         */
    uint8_t *Y_curr_dev;    /* current  luma plane (pitched)         */
    uint8_t *Y_out_dev;     /* denoised output (pitched)             */
    uint8_t *UV_dev;        /* current chroma plane (pitched)        */
    float   *flow_x;        /* optical flow x component              */
    float   *flow_y;        /* optical flow y component              */
    float   *u_est_dev;     /* chroma U IIR state (w/2 × h/2)       */
    float   *v_est_dev;     /* chroma V IIR state                    */
    int      width, height;
    int      y_pitch;
    int      uv_pitch;
    cudaStream_t stream;
    bool     initialized;
    bool     first_frame;
};

static NlmState g  = {};
static CUcontext g_cu = nullptr;
static int g_frame = 0;

/* ------------------------------------------------------------------ */
/* Error helpers                                                       */
/* ------------------------------------------------------------------ */
#define CUK(expr) do { \
    cudaError_t _e = (expr); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "[NLM] CUDA error %s at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
    } \
} while(0)

/* ================================================================== */
/* KERNEL 1 – Lucas-Kanade Optical Flow (per-pixel, 3×3 neighbourhood)*/
/* ================================================================== */
__global__ void k_lk_flow(
        const uint8_t * __restrict__ Y_prev,
        const uint8_t * __restrict__ Y_curr,
        float         * __restrict__ flow_x,
        float         * __restrict__ flow_y,
        int w, int h, int pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    /* Accumulate A^T A and A^T b over 3×3 window */
    float A11 = 0, A12 = 0, A22 = 0, b1 = 0, b2 = 0;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int cx = min(max(x + dx, 0), w - 1);
            int cy = min(max(y + dy, 0), h - 1);
            int rx = min(cx + 1, w - 1);
            int ry = min(cy + 1, h - 1);

            float Ix = (float)Y_curr[cy * pitch + rx]
                     - (float)Y_curr[cy * pitch + cx];
            float Iy = (float)Y_curr[ry * pitch + cx]
                     - (float)Y_curr[cy * pitch + cx];
            float It = (float)Y_curr[cy * pitch + cx]
                     - (float)Y_prev[cy * pitch + cx];

            A11 += Ix * Ix;
            A12 += Ix * Iy;
            A22 += Iy * Iy;
            b1  -= Ix * It;
            b2  -= Iy * It;
        }
    }

    A11 += 1e-3f;
    A22 += 1e-3f;
    float det = A11 * A22 - A12 * A12;

    float vx = 0.f, vy = 0.f;
    if (fabsf(det) > 1e-5f) {
        vx = (A22 * b1 - A12 * b2) / det;
        vy = (A11 * b2 - A12 * b1) / det;
    }

    /* Clamp to ±8 px */
    float mag = sqrtf(vx * vx + vy * vy);
    if (mag > 8.f) { float s = 8.f / mag; vx *= s; vy *= s; }

    int idx = y * w + x;
    flow_x[idx] = vx;
    flow_y[idx] = vy;
}

/* ================================================================== */
/* KERNEL 2 – Motion-Compensated Non-Local Means (luma)               */
/*                                                                     */
/* For each output pixel p we:                                         */
/*   a) shift the search centre by the flow vector  (motion compensation) */
/*   b) compare patches of size (2*PATCH_RADIUS+1)^2                  */
/*   c) weight candidates by Gaussian of patch SSD / h^2              */
/*   d) output normalised weighted sum                                 */
/* ================================================================== */
__global__ void k_nlm_denoise(
        const uint8_t * __restrict__ Y_in,
        float         * __restrict__ flow_x,
        float         * __restrict__ flow_y,
        uint8_t       * __restrict__ Y_out,
        int w, int h, int pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    /* Motion-compensated reference centre */
    int idx  = y * w + x;
    float cx = (float)x - flow_x[idx];
    float cy = (float)y - flow_y[idx];
    int   cxi = min(max((int)roundf(cx), 0), w - 1);
    int   cyi = min(max((int)roundf(cy), 0), h - 1);

    float h2   = c_NLM_H * c_NLM_H;
    int   pr   = c_PATCH_RADIUS;
    int   sr   = c_SEARCH_RADIUS;
    float wsum = 0.f;
    float vsum = 0.f;

    /* Iterate over search window centred at motion-compensated position */
    for (int qy = cyi - sr; qy <= cyi + sr; qy++) {
        for (int qx = cxi - sr; qx <= cxi + sr; qx++) {

            int qxc = min(max(qx, 0), w - 1);
            int qyc = min(max(qy, 0), h - 1);

            /* Patch SSD between pixel p and candidate q */
            float ssd = 0.f;
            int   cnt = 0;
            for (int py = -pr; py <= pr; py++) {
                for (int px = -pr; px <= pr; px++) {
                    int ax = min(max(x   + px, 0), w - 1);
                    int ay = min(max(y   + py, 0), h - 1);
                    int bx = min(max(qxc + px, 0), w - 1);
                    int by = min(max(qyc + py, 0), h - 1);

                    float diff = (float)Y_in[ay * pitch + ax]
                               - (float)Y_in[by * pitch + bx];
                    ssd += diff * diff;
                    cnt++;
                }
            }
            ssd /= (float)(cnt + 1);

            float w_ij = __expf(-ssd / h2);
            wsum += w_ij;
            vsum += w_ij * (float)Y_in[qyc * pitch + qxc];
        }
    }

    float result = (wsum > 1e-6f) ? (vsum / wsum) : (float)Y_in[y * pitch + x];
    result = fminf(fmaxf(result, 0.f), 255.f);
    Y_out[y * pitch + x] = (uint8_t)(result + 0.5f);
}

/* ================================================================== */
/* KERNEL 3 – Temporal blend between NLM output and raw frame         */
/*            (for static areas: trust NLM more; motion: trust raw)   */
/* ================================================================== */
__global__ void k_temporal_blend(
        const uint8_t * __restrict__ Y_nlm,
        const uint8_t * __restrict__ Y_raw,
        const float   * __restrict__ flow_x,
        const float   * __restrict__ flow_y,
        uint8_t       * __restrict__ Y_out,
        int w, int h, int pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int idx = y * w + x;
    float mag = sqrtf(flow_x[idx]*flow_x[idx] + flow_y[idx]*flow_y[idx]);

    /* alpha: how much we trust the NLM output                         */
    /* Static area → high trust (near c_TEMPORAL_BLEND); moving → less */
    float alpha = c_TEMPORAL_BLEND / (1.f + mag * 0.4f);
    alpha = fminf(fmaxf(alpha, 0.05f), c_TEMPORAL_BLEND);

    float v = alpha * (float)Y_nlm[y * pitch + x]
            + (1.f - alpha) * (float)Y_raw[y * pitch + x];
    Y_out[y * pitch + x] = (uint8_t)(fminf(fmaxf(v, 0.f), 255.f) + 0.5f);
}

/* ================================================================== */
/* KERNEL 4 – IIR chroma smoother (U/V interleaved NV12)              */
/* ================================================================== */
__global__ void k_chroma_iir(
        const uint8_t * __restrict__ UV_in,
        float         * __restrict__ u_est,
        float         * __restrict__ v_est,
        uint8_t       * __restrict__ UV_out,
        int wc, int hc, int pitch_c)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= wc || y >= hc) return;

    int flat   = y * wc + x;
    int base   = y * pitch_c + 2 * x;

    float U = (float)UV_in[base];
    float V = (float)UV_in[base + 1];

    float a = c_CHROMA_ALPHA;
    float u_new = a * U + (1.f - a) * u_est[flat];
    float v_new = a * V + (1.f - a) * v_est[flat];

    u_est[flat] = u_new;
    v_est[flat] = v_new;

    UV_out[base]     = (uint8_t)(fminf(fmaxf(u_new, 0.f), 255.f) + 0.5f);
    UV_out[base + 1] = (uint8_t)(fminf(fmaxf(v_new, 0.f), 255.f) + 0.5f);
}

/* ================================================================== */
/* Host helpers                                                        */
/* ================================================================== */

static void ensure_cuda_ctx()
{
    if (g_cu) return;
    cuInit(0);
    cuCtxGetCurrent(&g_cu);
    if (!g_cu) {
        CUdevice dev;
        cuDeviceGet(&dev, 0);
        cuCtxCreate(&g_cu, 0, dev);
    }
    cuCtxPushCurrent(g_cu);
}

static void setup_constants()
{
    float fv; int iv;
    fv = 12.f;  CUK(cudaMemcpyToSymbol(c_NLM_H,         &fv, sizeof(float)));
    iv = 2;     CUK(cudaMemcpyToSymbol(c_PATCH_RADIUS,   &iv, sizeof(int)));
    iv = 6;     CUK(cudaMemcpyToSymbol(c_SEARCH_RADIUS,  &iv, sizeof(int)));
    fv = 1.0f;  CUK(cudaMemcpyToSymbol(c_MOTION_THRESH,  &fv, sizeof(float)));
    fv = 0.20f; CUK(cudaMemcpyToSymbol(c_CHROMA_ALPHA,   &fv, sizeof(float)));
    fv = 0.85f; CUK(cudaMemcpyToSymbol(c_TEMPORAL_BLEND, &fv, sizeof(float)));
}

static void init_state(int w, int h, int pitch_y)
{
    g.width    = w;
    g.height   = h;
    g.y_pitch  = pitch_y;
    g.uv_pitch = pitch_y;

    size_t ysz   = (size_t)pitch_y * h;
    size_t uvsz  = (size_t)pitch_y * (h / 2);
    size_t fsz   = (size_t)w * h * sizeof(float);
    size_t csz   = (size_t)(w / 2) * (h / 2) * sizeof(float);

    CUK(cudaMalloc(&g.Y_prev_dev, ysz));
    CUK(cudaMalloc(&g.Y_curr_dev, ysz));
    CUK(cudaMalloc(&g.Y_out_dev,  ysz));
    CUK(cudaMalloc(&g.UV_dev,     uvsz));
    CUK(cudaMalloc(&g.flow_x,     fsz));
    CUK(cudaMalloc(&g.flow_y,     fsz));
    CUK(cudaMalloc(&g.u_est_dev,  csz));
    CUK(cudaMalloc(&g.v_est_dev,  csz));

    CUK(cudaMemset(g.flow_x, 0, fsz));
    CUK(cudaMemset(g.flow_y, 0, fsz));

    /* Initialise chroma IIR state to neutral grey (128) */
    size_t nel = (size_t)(w / 2) * (h / 2);
    float *tmp = (float*)malloc(csz);
    for (size_t i = 0; i < nel; i++) tmp[i] = 128.f;
    CUK(cudaMemcpy(g.u_est_dev, tmp, csz, cudaMemcpyHostToDevice));
    CUK(cudaMemcpy(g.v_est_dev, tmp, csz, cudaMemcpyHostToDevice));
    free(tmp);

    CUK(cudaStreamCreate(&g.stream));
    g.initialized = true;
    g.first_frame = true;

    fprintf(stderr, "[NLM] init_state: w=%d h=%d pitch=%d\n", w, h, pitch_y);
}

/* ------------------------------------------------------------------ */
/* Pull planes from EGL frame into our device buffers                  */
/* ------------------------------------------------------------------ */
static void import_frame(CUeglFrame *f, int w, int h)
{
    if (f->frameType == CU_EGL_FRAME_TYPE_ARRAY) {
        cudaArray_t aY  = reinterpret_cast<cudaArray_t>(f->frame.pArray[0]);
        cudaArray_t aUV = reinterpret_cast<cudaArray_t>(f->frame.pArray[1]);

        CUK(cudaMemcpy2DFromArray(g.Y_curr_dev, g.y_pitch,
                                  (cudaArray_const_t)aY, 0, 0,
                                  w, h, cudaMemcpyDeviceToDevice));
        CUK(cudaMemcpy2DFromArray(g.UV_dev, g.uv_pitch,
                                  (cudaArray_const_t)aUV, 0, 0,
                                  w, h / 2, cudaMemcpyDeviceToDevice));
    } else {
        uint8_t *py  = (uint8_t*)f->frame.pPitch[0];
        uint8_t *puv = (uint8_t*)f->frame.pPitch[1];

        CUK(cudaMemcpy2D(g.Y_curr_dev, g.y_pitch,
                         py, f->pitch, w, h, cudaMemcpyDeviceToDevice));
        CUK(cudaMemcpy2D(g.UV_dev, g.uv_pitch,
                         puv, f->pitch, w, h / 2, cudaMemcpyDeviceToDevice));
    }
}

/* Push denoised planes back into EGL frame */
static void export_frame(CUeglFrame *f, int w, int h)
{
    if (f->frameType == CU_EGL_FRAME_TYPE_ARRAY) {
        cudaArray_t aY  = reinterpret_cast<cudaArray_t>(f->frame.pArray[0]);
        cudaArray_t aUV = reinterpret_cast<cudaArray_t>(f->frame.pArray[1]);

        CUK(cudaMemcpy2DToArray(aY,  0, 0,
                                g.Y_out_dev, g.y_pitch,
                                w, h, cudaMemcpyDeviceToDevice));
        CUK(cudaMemcpy2DToArray(aUV, 0, 0,
                                g.UV_dev, g.uv_pitch,
                                w, h / 2, cudaMemcpyDeviceToDevice));
    } else {
        uint8_t *py  = (uint8_t*)f->frame.pPitch[0];
        uint8_t *puv = (uint8_t*)f->frame.pPitch[1];

        CUK(cudaMemcpy2D(py, f->pitch,
                         g.Y_out_dev, g.y_pitch,
                         w, h, cudaMemcpyDeviceToDevice));
        CUK(cudaMemcpy2D(puv, f->pitch,
                         g.UV_dev, g.uv_pitch,
                         w, h / 2, cudaMemcpyDeviceToDevice));
    }
}

/* ------------------------------------------------------------------ */
/* Main per-frame processing                                            */
/* ------------------------------------------------------------------ */
static void process_frame(CUeglFrame *f)
{
    int w = (int)f->width;
    int h = (int)f->height;
    int pitch_y = (f->frameType == CU_EGL_FRAME_TYPE_PITCH)
                ? (int)f->pitch : w;

    fprintf(stderr, "[NLM] frame=%d w=%d h=%d frameType=%d\n",
            g_frame, w, h, (int)f->frameType);

    if (!g.initialized) {
        init_state(w, h, pitch_y);
        setup_constants();
    }

    import_frame(f, w, h);
    CUK(cudaStreamSynchronize(g.stream));

    dim3 thr(16, 16);
    dim3 blk((w + 15) / 16, (h + 15) / 16);
    dim3 blk_uv((w / 2 + 15) / 16, (h / 2 + 15) / 16);

    if (g.first_frame) {
        /* First frame: copy raw to output, no denoising reference yet */
        CUK(cudaMemcpy2DAsync(g.Y_out_dev, g.y_pitch,
                              g.Y_curr_dev, g.y_pitch,
                              w, h, cudaMemcpyDeviceToDevice, g.stream));
        g.first_frame = false;
    } else {
        /* Step 1 – Optical flow: Y_prev → Y_curr */
        k_lk_flow<<<blk, thr, 0, g.stream>>>(
                g.Y_prev_dev, g.Y_curr_dev,
                g.flow_x, g.flow_y,
                w, h, g.y_pitch);
        CUK(cudaGetLastError());

        /* Step 2 – Motion-compensated NLM on luma → Y_out */
        k_nlm_denoise<<<blk, thr, 0, g.stream>>>(
                g.Y_curr_dev, g.flow_x, g.flow_y,
                g.Y_out_dev,
                w, h, g.y_pitch);
        CUK(cudaGetLastError());

        /* Step 3 – Temporal blend: trust NLM more in static regions */
        k_temporal_blend<<<blk, thr, 0, g.stream>>>(
                g.Y_out_dev, g.Y_curr_dev,
                g.flow_x, g.flow_y,
                g.Y_out_dev,
                w, h, g.y_pitch);
        CUK(cudaGetLastError());

        /* Step 4 – IIR chroma filter */
        k_chroma_iir<<<blk_uv, thr, 0, g.stream>>>(
                g.UV_dev, g.u_est_dev, g.v_est_dev,
                g.UV_dev,
                w / 2, h / 2, g.uv_pitch);
        CUK(cudaGetLastError());
    }

    export_frame(f, w, h);

    /* Keep previous luma for next iteration */
    CUK(cudaMemcpy2DAsync(g.Y_prev_dev, g.y_pitch,
                          g.Y_curr_dev, g.y_pitch,
                          w, h, cudaMemcpyDeviceToDevice, g.stream));
    CUK(cudaStreamSynchronize(g.stream));
}

/* ------------------------------------------------------------------ */
/* EGL entry point                                                      */
/* ------------------------------------------------------------------ */
static void cuda_process(EGLImageKHR image, void**)
{
    ++g_frame;
    if (g_frame == 1) ensure_cuda_ctx();

    fprintf(stderr, "[NLM] cuda_process frame=%d image=%p\n",
            g_frame, (void*)image);

    CUgraphicsResource res = nullptr;
    CUeglFrame eglFrame;

    if (cuGraphicsEGLRegisterImage(&res, image,
                CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE) != CUDA_SUCCESS) {
        fprintf(stderr, "[NLM] cuGraphicsEGLRegisterImage failed\n");
        return;
    }
    if (cuGraphicsResourceGetMappedEglFrame(&eglFrame, res, 0, 0)
            != CUDA_SUCCESS) {
        fprintf(stderr, "[NLM] cuGraphicsResourceGetMappedEglFrame failed\n");
        cuGraphicsUnregisterResource(res);
        return;
    }

    process_frame(&eglFrame);
    cuGraphicsUnregisterResource(res);
}

/* ------------------------------------------------------------------ */
/* Cleanup                                                              */
/* ------------------------------------------------------------------ */
static void cleanup()
{
    if (!g.initialized) return;
    if (g.Y_prev_dev)  cudaFree(g.Y_prev_dev);
    if (g.Y_curr_dev)  cudaFree(g.Y_curr_dev);
    if (g.Y_out_dev)   cudaFree(g.Y_out_dev);
    if (g.UV_dev)      cudaFree(g.UV_dev);
    if (g.flow_x)      cudaFree(g.flow_x);
    if (g.flow_y)      cudaFree(g.flow_y);
    if (g.u_est_dev)   cudaFree(g.u_est_dev);
    if (g.v_est_dev)   cudaFree(g.v_est_dev);
    if (g.stream)      cudaStreamDestroy(g.stream);
    memset(&g, 0, sizeof(g));
}

/* ------------------------------------------------------------------ */
/* Public init (called by GStreamer plugin)                             */
/* ------------------------------------------------------------------ */
extern "C" void init(CustomerFunction *f)
{
    f->fPreProcess  = nullptr;
    f->fGPUProcess  = cuda_process;
    f->fPostProcess = nullptr;
    atexit(cleanup);
}
