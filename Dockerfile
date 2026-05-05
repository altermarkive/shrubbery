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
        libopenblas-dev \
        pkg-config \
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
# libopenblas-dev - N/A
# pkg-config - matplotlib
# unzip - N/A
# zlib - Pillow

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_PYTHON_INSTALL_DIR="/opt/uv-python"
ENV UV_NO_CACHE=1
ENV VIRTUAL_ENV="/shrubbery/.venv"
ADD . /shrubbery
RUN cd /shrubbery && UV_HTTP_TIMEOUT=60 /bin/uv sync --locked && \
    chmod -R a+rX $VIRTUAL_ENV $UV_PYTHON_INSTALL_DIR
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONPATH="$VIRTUAL_ENV/lib/python3.13/site-packages:/shrubbery/src"

# TensorRT is quite noisy
ENV PYTHONWARNINGS="ignore::SyntaxWarning"

ENTRYPOINT [ "/bin/bash" ]

# TODO: Follow the best practices listed at https://docs.astral.sh/uv/guides/integration/docker/ and switch to regular user
