# ds-nlm-denoise — Makefile
# Targets:  Jetson Orin (sm_87),  DeepStream 7.1,  CUDA 12.x
#
# Usage:
#   make CUDA_VER=12.6
#   make CUDA_VER=12.6 install
#   make clean

CUDA_VER ?=
ifeq ($(CUDA_VER),)
  $(error CUDA_VER is not set. Example: make CUDA_VER=12.6)
endif

NVDS_VERSION ?= 7.1

CUDA_HOME  := /usr/local/cuda-$(CUDA_VER)
NVDS_ROOT  := /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)

# ── Sources ──────────────────────────────────────────────────────────
SRCS_CPP := gst_ds_nlm_denoise.cpp
SRCS_CU  := nlm_denoise.cu
INCS     := gst_ds_nlm_denoise.h customer_functions.h

LIB      := libnvdsgst_ds_nlm_denoise.so

# ── Install dirs ─────────────────────────────────────────────────────
GST_INSTALL_DIR ?= $(NVDS_ROOT)/lib/gst-plugins
LIB_INSTALL_DIR ?= $(NVDS_ROOT)/lib

# ── C++ compiler ─────────────────────────────────────────────────────
CXX    := g++
PKGS   := gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0

CFLAGS := -fPIC \
          -DDS_VERSION=\"$(NVDS_VERSION).0\" \
          -I$(CUDA_HOME)/include \
          -I$(NVDS_ROOT)/sources/includes \
          -Wno-deprecated-declarations \
          $(shell pkg-config --cflags $(PKGS))

LIBS   := -shared -Wl,-no-undefined \
          -L$(CUDA_HOME)/lib64 -lcudart -lcuda -ldl \
          -L$(LIB_INSTALL_DIR) \
            -lnvdsgst_helper \
            -lnvdsgst_meta \
            -lnvds_meta \
            -lnvbufsurface \
            -lnvbufsurftransform \
          $(shell pkg-config --libs $(PKGS)) \
          -Wl,-rpath,$(LIB_INSTALL_DIR)

# ── NVCC ─────────────────────────────────────────────────────────────
NVCC       := $(CUDA_HOME)/bin/nvcc
# sm_87 = Ampere, covers all Jetson Orin variants
# Adjust to sm_90 for Orin NX / AGX Orin with CUDA 12.4+
NVCC_FLAGS := -m64 \
              -gencode arch=compute_87,code=sm_87 \
              -I$(CUDA_HOME)/include \
              -I$(NVDS_ROOT)/sources/includes \
              -Xcompiler -fPIC \
              --std=c++14

# ── Object files ─────────────────────────────────────────────────────
OBJS := $(SRCS_CPP:.cpp=.o) $(SRCS_CU:.cu=.cu.o)

# ── Rules ─────────────────────────────────────────────────────────────
.PHONY: all install clean

all: $(LIB)

%.o: %.cpp $(INCS) Makefile
	@echo "[CXX]  $<"
	$(CXX) $(CFLAGS) -c -o $@ $<

%.cu.o: %.cu Makefile
	@echo "[NVCC] $<"
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(LIB): $(OBJS) Makefile
	@echo "[LD]   $@"
	$(CXX) -o $@ $(OBJS) $(LIBS)

install: $(LIB)
	@echo "Installing $(LIB) → $(GST_INSTALL_DIR)"
	install -m 755 $(LIB) $(GST_INSTALL_DIR)

clean:
	@echo "Cleaning build artifacts"
	rm -f $(OBJS) $(LIB)
