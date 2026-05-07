# Sync NVIDIA Driver & CUDA versions (`nvidia-smi`)
FROM nvidia/cuda:13.2.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -qy update \
    && apt-get -qy remove python \
    && apt-get -qyy install \
        -o APT::Install-Recommends=false \
        -o APT::Install-Suggests=false \
        build-essential \
        cmake \
        curl \
        git \
        git-lfs \
        libatlas3-base \
        libblas-dev \
        libboost-dev \
        libboost-system-dev \
        libboost-filesystem-dev \
        libnvidia-compute-595-server \
        libopenblas-dev \
        nvidia-opencl-dev \
        opencl-headers \
        pkg-config \
        unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# build-essential - N/A
# cmake - lightgbm
# curl - N/A
# git - N/A
# git-lfs - N/A
# libatlas3-base - N/A
# libblas-dev - N/A
# libboost-dev - lightgbm
# libboost-system-dev - lightgbm
# libboost-filesystem-dev - lightgbm
# libnvidia-compute-595-server
# libopenblas-dev - N/A
# nvidia-opencl-dev - lightgbm
# opencl-headers - lightgbm
# unzip - N/A

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_HTTP_TIMEOUT=60 \
    UV_PYTHON=python3.13 \
    UV_PYTHON_INSTALL_DIR=/app/python \
    VIRTUAL_ENV=/app/venv
WORKDIR /app
RUN echo 'cmake_language(DEFER CALL set_target_properties lightgbm_objs _lightgbm lightgbm_capi_objs PROPERTIES CUDA_ARCHITECTURES "80;86;89;90;100;120")' \
    > /tmp/lgbm_cuda_archs.cmake
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    /bin/uv sync --active --frozen --no-dev --no-install-project \
        --no-binary-package lightgbm \
        --config-settings "cmake.define.USE_GPU=ON" \
        --config-settings "cmake.define.USE_CUDA=ON" \
        --config-settings "cmake.define.CMAKE_PROJECT_INCLUDE=/tmp/lgbm_cuda_archs.cmake"
COPY . /app/package
WORKDIR /app/package
RUN --mount=type=cache,target=/root/.cache/uv \
    /bin/uv pip install . && \
    rm -rf /app/package
WORKDIR /app
RUN chmod -R a+rX /app
ENV PATH="/app/venv/bin:$PATH"

# TensorRT is quite noisy
ENV PYTHONWARNINGS="ignore::SyntaxWarning"

ENTRYPOINT [ "/bin/bash" ]
