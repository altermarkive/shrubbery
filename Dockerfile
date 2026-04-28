# Sync NVIDIA Driver & CUDA versions (`nvidia-smi`)
FROM nvidia/cuda:13.2.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -yq update && \
    apt-get -yq remove python && \
    apt-get -yq install \
        build-essential \
        cmake \
        curl \
        git-lfs \
        libatlas3-base \
        libblas-dev \
        libfreetype-dev \
        libjpeg-dev \
        libnvidia-compute-595-server \
        libopenblas-dev \
        pkg-config \
        software-properties-common \
        unzip \
        zlib1g-dev
# build-essential - tornado
# cmake - lightgbm
# curl - uv
# git-lfs - N/A
# libatlas3-base - N/A
# libblas-dev - N/A
# libfreetype-dev - matplotlib
# libjpeg-dev - Pillow
# libnvidia-compute-595-server - lightgbm
# libopenblas-dev - N/A
# pkg-config - matplotlib
# software-properties-common - Python
# unzip - N/A
# zlib - Pillow

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get -yq install python3.13-full python3.13-dev && \
    python3 --version && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1 && \
    update-alternatives --config python3 && \
    python3 --version && \
    /usr/bin/python3.13 -m ensurepip --upgrade

RUN /usr/bin/python3.13 -m pip install --disable-pip-version-check --no-cache-dir --break-system-packages uv

ADD . /tmp/shrubbery
RUN cd /tmp/shrubbery && UV_HTTP_TIMEOUT=60 uv pip install --system '.[dev]' && rm -rf /tmp/shrubbery

# TensorRT is quite noisy
ENV PYTHONWARNINGS="ignore::SyntaxWarning"

ENTRYPOINT [ "/bin/bash" ]

# TODO: Follow the best practices listed at https://docs.astral.sh/uv/guides/integration/docker/ and switch to regular user
