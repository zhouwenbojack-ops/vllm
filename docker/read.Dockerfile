FROM ubuntu:24.04

ARG DEV_USER=developer
ARG DEV_UID=1000
ARG DEV_GID=1000
ARG PYTHON_VERSION=3.12
ARG UV_DEFAULT_INDEX=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
ARG UV_PYTHON_INSTALL_MIRROR=https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone/

# Docker treats these as predefined build arguments and excludes their values
# from the image history.
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG ALL_PROXY
ARG NO_PROXY
ARG http_proxy
ARG https_proxy
ARG all_proxy
ARG no_proxy

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    HOME=/home/${DEV_USER} \
    USER=${DEV_USER} \
    LOGNAME=${DEV_USER} \
    PATH=/opt/uv/bin:${PATH} \
    UV_INSTALL_DIR=/opt/uv/bin \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_CACHE_DIR=/home/${DEV_USER}/.cache/uv \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=500 \
    UV_INDEX_STRATEGY=unsafe-best-match \
    UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX} \
    UV_PYTHON_INSTALL_MIRROR=${UV_PYTHON_INSTALL_MIRROR} \
    CCACHE_DIR=/home/${DEV_USER}/.cache/ccache \
    CCACHE_MAXSIZE=20G \
    CMAKE_C_COMPILER_LAUNCHER=ccache \
    CMAKE_CXX_COMPILER_LAUNCHER=ccache \
    VLLM_TARGET_DEVICE=cpu

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        bash-completion \
        bat \
        build-essential \
        ca-certificates \
        ccache \
        clang \
        clang-format \
        clangd \
        curl \
        fd-find \
        ffmpeg \
        fzf \
        g++-12 \
        gcc-12 \
        gdb \
        git \
        jq \
        less \
        libgl1 \
        libnuma-dev \
        libsm6 \
        libtcmalloc-minimal4 \
        libxext6 \
        lsof \
        make \
        ninja-build \
        openssh-client \
        pkg-config \
        python3-dev \
        ripgrep \
        sudo \
        tmux \
        tree \
        vim \
        xz-utils \
        zlib1g-dev \
        zoxide \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 120 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-12 \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    group_name="$(getent group "${DEV_GID}" | cut -d: -f1 || true)"; \
    if [ -z "${group_name}" ]; then \
        group_name="${DEV_USER}"; \
        groupadd --gid "${DEV_GID}" "${group_name}"; \
    fi; \
    useradd --create-home --shell /bin/bash --uid "${DEV_UID}" \
        --gid "${group_name}" "${DEV_USER}"; \
    install -d -o "${DEV_UID}" -g "${DEV_GID}" \
        "/home/${DEV_USER}/.cache/ccache" \
        "/home/${DEV_USER}/.cache/uv" \
        /opt/uv; \
    echo "${DEV_USER} ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/${DEV_USER}"; \
    chmod 0440 "/etc/sudoers.d/${DEV_USER}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && test "$(python3 --version | cut -d ' ' -f 2 | cut -d. -f1,2)" = "${PYTHON_VERSION}" \
    && python3 -m pip install \
        --break-system-packages \
        --no-cache-dir \
        --index-url "${UV_DEFAULT_INDEX}" \
        uv \
    && uv --version \
    && chown -R "${DEV_UID}:${DEV_GID}" /opt/uv

COPY --chown=${DEV_UID}:${DEV_GID} docker/read.bashrc.template \
    /home/${DEV_USER}/.bashrc

USER ${DEV_USER}
WORKDIR /workspace

CMD ["sleep", "infinity"]
