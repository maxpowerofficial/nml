# ds-nlm-denoise

**CUDA Non-Local Means (NLM) video denoiser with Lucas-Kanade Optical Flow**  
_DeepStream 7.1 · Jetson Orin · GStreamer plugin_

---

## Overview

`ds-nlm-denoise` is a GStreamer in-place transform plugin for **NVIDIA DeepStream 7.1** that denoises video frames entirely on the GPU using:

| Component | Algorithm |
|-----------|-----------|
| Motion estimation | Lucas-Kanade Optical Flow (per-pixel, 3×3 window) |
| Luma denoising | Motion-Compensated Non-Local Means (NLM) |
| Temporal stability | Adaptive temporal blend (motion-aware alpha) |
| Chroma denoising | Per-pixel IIR low-pass on U and V channels |

All kernels run on CUDA, operate in-place on `NvBufSurface` memory (NVMM), and target **Jetson Orin (sm_87)**.

---

## Why NLM + Optical Flow?

NLM (Non-Local Means) is a reference-grade denoising algorithm from _Buades, Coll & Morel (2005)_, widely used as a quality benchmark in image processing. Its core idea: instead of averaging only neighbouring pixels, it averages **all pixels in the frame weighted by patch similarity**. This makes it exceptionally good at preserving fine texture and sharp edges while eliminating noise.

Combining NLM with Lucas-Kanade Optical Flow allows the patch search window to be motion-compensated — moving objects are not smeared or ghosted because the search centre tracks where each pixel actually moved.

| Property | Value |
|---|---|
| Model assumption | None — pure patch similarity |
| Edge preservation | Excellent (patches respect texture boundaries) |
| Uniform regions | Strong smoothing |
| Fine texture / detail | Preserved via patch matching |
| Tuning | Single `h` parameter (filter strength) |
| Motion handling | Lucas-Kanade flow warps search window per-pixel |

---

## Algorithm Details

### 1. Lucas-Kanade Optical Flow (`k_lk_flow`)

For every pixel `(x, y)` we solve the 2×2 normal equation:

```
A^T A · v = A^T b
```

where `A` accumulates spatial gradients `[Ix, Iy]` and `b` accumulates temporal differences `It` over a 3×3 neighbourhood.  Flow is clamped to ±8 px/frame.

### 2. Motion-Compensated NLM (`k_nlm_denoise`)

Output pixel `y(p)` is a weighted average:

```
y(p) = Σ_q  w(p,q) · I(q) / Σ_q w(p,q)

w(p,q) = exp( -||I(Np) - I(Nq)||² / h² )
```

where `Np`, `Nq` are patch neighbourhoods and `h` is the filter strength.  
The centre of the search window for `p` is shifted by the optical flow vector, so moving edges are not smeared.

### 3. Temporal Blend (`k_temporal_blend`)

```
α(p) = TEMPORAL_BLEND / (1 + |v(p)| · 0.4)
out(p) = α · nlm(p) + (1 − α) · raw(p)
```

Static pixels trust NLM more; moving pixels retain more of the raw signal to avoid ghosting.

### 4. Chroma IIR (`k_chroma_iir`)

```
U_est[t] = α · U_raw[t] + (1 − α) · U_est[t-1]
```

Simple exponential smoothing on U/V channels with `α = CHROMA_ALPHA` (default 0.20).

---

## Requirements

| Dependency | Version |
|---|---|
| Jetson Orin (AGX / NX / Nano) | JetPack 6.x |
| CUDA | 12.x |
| DeepStream | **7.1** |
| GStreamer | 1.20+ (bundled with DS 7.1) |
| GLib / pkg-config | system packages |

---

## Build

```bash
# Clone the repo
git clone https://github.com/your-org/ds-nlm-denoise.git
cd ds-nlm-denoise

# Build (replace with your CUDA version)
make CUDA_VER=12.6

# Install to DeepStream plugin directory
sudo make CUDA_VER=12.6 install
```

The resulting shared library is `libnvdsgst_ds_nlm_denoise.so`.

---

## GStreamer Plugin Usage

### Register / verify the plugin

```bash
# Refresh GStreamer plugin cache
export GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream-7.1/lib/gst-plugins
gst-inspect-1.0 ds_nlm_denoise
```

### Minimal pipeline — USB camera → NLM denoise → display

```bash
gst-launch-1.0 \
  v4l2src device=/dev/video0 ! \
  "video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1" ! \
  nvvideoconvert ! \
  "video/x-raw(memory:NVMM),format=NV12" ! \
  ds_nlm_denoise enable-denoise=true ! \
  nvvideoconvert ! \
  nv3dsink
```

### RTSP stream → NLM denoise → re-encode → file

```bash
gst-launch-1.0 \
  rtspsrc location=rtsp://192.168.1.10/stream ! \
  rtph264depay ! h264parse ! \
  nvv4l2decoder ! \
  "video/x-raw(memory:NVMM),format=NV12" ! \
  ds_nlm_denoise enable-denoise=true gpu-id=0 ! \
  nvv4l2h264enc ! \
  h264parse ! mp4mux ! \
  filesink location=denoised_output.mp4
```

### Inside a DeepStream pipeline (with nvinfer)

```bash
gst-launch-1.0 \
  filesrc location=noisy_video.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! \
  nvstreammux name=mux batch-size=1 width=1920 height=1080 ! \
  ds_nlm_denoise enable-denoise=true batch-size=1 ! \
  nvinfer config-file-path=config_infer_primary.txt ! \
  nvvideoconvert ! \
  nvdsosd ! \
  nv3dsink
```

### Disable denoiser at runtime (passthrough)

```bash
# Set enable-denoise=false to bypass without stopping the pipeline
gst-launch-1.0 \
  ... ! ds_nlm_denoise enable-denoise=false ! ...
```

---

## Plugin Properties

| Property | Type | Default | Description |
|---|---|---|---|
| `unique-id` | uint | 20 | Unique element identifier for metadata tagging |
| `processing-width` | int | 1920 | Processing width |
| `processing-height` | int | 1080 | Processing height |
| `enable-denoise` | bool | true | Enable / disable NLM denoiser |
| `gpu-id` | uint | 0 | GPU device to run kernels on |
| `batch-size` | uint | 1 | Maximum batch size |

---

## Tunable CUDA Constants

These are compile-time constants in `nlm_denoise.cu` loaded into CUDA constant memory via `setup_constants()`.  Edit and recompile to tune.

| Constant | Default | Effect |
|---|---|---|
| `c_NLM_H` | 12.0 | Filter strength. ↑ = more smoothing, ↓ = preserve details |
| `c_PATCH_RADIUS` | 2 | Patch half-size (5×5 patches) |
| `c_SEARCH_RADIUS` | 6 | Search window half-size (13×13) |
| `c_MOTION_THRESH` | 1.0 px | Not used for switching in NLM (retained for future use) |
| `c_CHROMA_ALPHA` | 0.20 | IIR coefficient for chroma (0 = no update, 1 = no smoothing) |
| `c_TEMPORAL_BLEND` | 0.85 | Max NLM trust weight in static regions |

> **Performance tip:** Default settings (`PATCH_RADIUS=2`, `SEARCH_RADIUS=6`) deliver **160 FPS at 1080p** on Jetson AGX Orin. For even higher throughput reduce `SEARCH_RADIUS` to 4 — quality stays excellent and FPS rises to ~260.

---

## Performance (Jetson AGX Orin)

The plugin sustains **160 FPS** at 1080p on Jetson AGX Orin with the default configuration.

| Config | Resolution | FPS | GPU util | Latency/frame |
|---|---|---|---|---|
| Optical flow only | 1080p | >400 | ~5% | ~0.5 ms |
| NLM (3×3 patch, 9×9 search) | 1080p | ~260 | ~25% | ~3.8 ms |
| NLM (5×5 patch, 13×13 search) — **default** | 1080p | **160** | ~45% | ~6.2 ms |
| NLM (7×7 patch, 21×21 search) | 1080p | ~75 | ~80% | ~13 ms |
| NLM (5×5 patch, 13×13 search) — **default** | 4K | ~42 | ~70% | ~23 ms |

> Measured on Jetson AGX Orin 64 GB, JetPack 6.1, CUDA 12.6, clocks maxed (`sudo nvpmodel -m 0 && sudo jetson_clocks`).

---

## Project Structure

```
ds-nlm-denoise/
├── nlm_denoise.cu          # CUDA kernels: LK flow + NLM + temporal blend + chroma IIR
├── gst_ds_nlm_denoise.cpp  # GStreamer plugin boilerplate
├── gst_ds_nlm_denoise.h    # Plugin struct / class definitions
├── customer_functions.h    # NVIDIA CustomerFunction hook interface
├── Makefile                # Build system
└── README.md               # This file
```

---

## License

This project builds on NVIDIA DeepStream SDK sample code (NVIDIA Proprietary License) and the CUDA runtime.  The NLM algorithm implementation (`nlm_denoise.cu`) is original work and may be used freely under the terms of your agreement with NVIDIA DeepStream.

---

## References

- Buades, A., Coll, B., & Morel, J.-M. (2005). *A non-local algorithm for image denoising.* CVPR 2005.
- Lucas, B. D., & Kanade, T. (1981). *An iterative image registration technique with an application to stereo vision.* IJCAI 1981.
- NVIDIA DeepStream SDK 7.1 Developer Guide — https://docs.nvidia.com/metropolis/deepstream/dev-guide/
- NVIDIA Jetson Orin Technical Reference Manual — https://developer.nvidia.com/embedded/jetson-agx-orin
