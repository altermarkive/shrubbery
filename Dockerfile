# Sync NVIDIA Driver & CUDA versions (`nvidia-smi`)
FROM nvidia/cuda:13.2.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -yq update && \
    apt-get -yq remove python && \
    apt-get -yyq install \
        -o APT::Install-Recommends=false \
        -o APT::Install-Suggests=false \
        build-essential \
        cmake \
        curl \
        enscript \
        ffmpeg \
        git \
        git-lfs \
        imagemagick \
        jq \
        libatlas3-base \
        libblas-dev \
        libfreetype-dev \
        libhdf5-dev \
        libjpeg-dev \
        libasound2-plugins \
        libnvidia-compute-595-server \
        libopenblas-dev \
        ncurses-bin \
        pipewire-bin \
        pkg-config \
        poppler-utils \
        qpdf \
        ripgrep \
        software-properties-common \
        sudo \
        unzip \
        zlib1g-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
# build-essential - N/A
# cmake - lightgbm (older, non-wheel version)
# curl - N/A
# enscript - N/A
# ffmpeg - N/A
# git - N/A
# git-lfs - N/A
# imagemagick - N/A
# jq - N/A
# libatlas3-base - N/A
# libblas-dev - N/A
# libfreetype-dev - matplotlib (older, non-wheel version)
# libhdf5-dev - Keras
# libjpeg-dev - Pillow (older, non-wheel version)
# libasound2-plugins - Claude Code voice (ALSA → PulseAudio routing)
# libnvidia-compute-595-server - lightgbm
# libopenblas-dev - N/A
# ncurses-bin - Claude Code
# pipewire-bin - Claude Code
# pkg-config - matplotlib (older, non-wheel version)
# poppler-utils - N/A
# qpdf - N/A
# ripgrep - Claude Code
# software-properties-common - Python
# unzip - N/A
# zlib - Pillow (older, non-wheel version)

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_HTTP_TIMEOUT=60 \
    UV_PYTHON=python3.13 \
    UV_PYTHON_INSTALL_DIR=/app/python \
    VIRTUAL_ENV=/app/venv
# TensorRT is quite noisy
ENV PYTHONWARNINGS="ignore::SyntaxWarning"
# Initiate Virtual Environment
RUN /bin/uv venv

ENV USER=user
ENV HOME=/home/$USER
RUN userdel -r ubuntu 2> /dev/null || true
RUN useradd -m -s /bin/bash -u 1000 $USER && \
    chown -R $USER:$USER $HOME /app && \
    echo "$USER ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER $USER

ENV UV_NO_CACHE=1
RUN --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    /bin/uv sync --active --frozen --no-dev --no-install-project
COPY --chown=$USER:$USER . /app/shrubbery
WORKDIR /app/shrubbery
RUN /bin/uv pip install --python $VIRTUAL_ENV . && \
    rm -rf /app/shrubbery
WORKDIR /app
ENV PATH="$PATH:$VIRTUAL_ENV/bin"
