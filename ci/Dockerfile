FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops

ARG INSTALL_TEST_REQUIREMENTS=false

ENV DEBIAN_FRONTEND=noninteractive \
    ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest \
    ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest \
    LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

SHELL ["/bin/bash", "-lc"]

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      bash \
      ca-certificates \
      cmake \
      curl \
      g++ \
      gcc \
      git \
      git-lfs \
      make \
      ninja-build \
      openssh-client \
      pkg-config \
      python3-dev \
      python3-pip \
      python3-setuptools \
      python3-wheel \
      valgrind && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/fla-ci/requirements.txt
COPY tests/requirements.txt /tmp/fla-ci/tests-requirements.txt

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install -r /tmp/fla-ci/requirements.txt && \
    python3 -m pip install triton triton-ascend && \
    python3 -m pip install --target=/tmp/pybind11-target pybind11 && \
    cp -a /tmp/pybind11-target/. /usr/local/lib/python3.10/dist-packages/ && \
    rm -rf /tmp/pybind11-target && \
    python3 -c "import pybind11; print(pybind11.__file__)" && \
    if [[ "${INSTALL_TEST_REQUIREMENTS}" == "true" ]]; then \
      grep -v -E '^[[:space:]]*(#|$|valgrind[[:space:]]*$)' /tmp/fla-ci/tests-requirements.txt \
        | python3 -m pip install -r /dev/stdin; \
    fi

WORKDIR /workspace/repo
