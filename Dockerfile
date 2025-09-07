# Sync with NVIDIA Driver & CUDA versions (`nvidia-smi`)
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

# 1. NVIDIA CUDA Installation Guide for Linux (Ubuntu): https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu
# 2. Installing the NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt

ENV DEBIAN_FRONTEND=noninteractive

# Python package dependencies
RUN apt-get -yq update && \
    apt-get -yq remove python && \
    apt-get -yq install \
        build-essential \
        cmake \
        curl \
        git-lfs \
        libatlas3-base \
        libblas-dev \
        libfftw3-dev \
        libfreetype-dev \
        libhdf5-dev \
        libjpeg-dev \
        libnvidia-compute-570-server \
        libopenblas-dev \
        pkg-config \
        software-properties-common \
        unzip \
        zlib1g-dev
# build-essential - tornado
# cmake - lightgbm (older, non-wheel version)
# curl - uv
# git-lfs - N/A
# libatlas3-base - N/A
# libblas-dev - N/A
# libfftw3-dev - opentsne
# libfreetype-dev - matplotlib (older, non-wheel version)
# libhdf5-dev - Keras
# libjpeg-dev - Pillow (older, non-wheel version)
# libnvidia-compute-570-server - lightgbm
# libopenblas-dev - N/A
# pkg-config - matplotlib (older, non-wheel version)
# software-properties-common - Python
# unzip - N/A
# zlib - Pillow (older, non-wheel version)

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get -yq install python3.13-full python3.13-dev && \
    python3 --version && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1 && \
    update-alternatives --config python3 && \
    python3 --version && \
    /usr/bin/python3.13 -m ensurepip --upgrade

RUN /usr/bin/python3.13 -m pip install --disable-pip-version-check --no-cache-dir --break-system-packages uv

ADD . /tmp/shrubbery
RUN cd /tmp/shrubbery && uv pip install --system '.[dev]' && rm -rf /tmp/shrubbery

ENTRYPOINT [ "/usr/local/bin/shrubbery" ]

# TODO: Follow the best practices listed at https://docs.astral.sh/uv/guides/integration/docker/ and switch to regular user
