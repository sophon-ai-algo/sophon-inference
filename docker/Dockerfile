FROM ubuntu:18.04
ARG DEBIAN_FRONTEND=nointeractive
ENV TZ=Asia/Shanghai

RUN apt-get update
RUN apt-get install -y \
    cmake \
    libtool \
    wget \
    git \
	vim \
    tree \
    libssl-dev \
    pkg-config libopencv-dev libncurses5-dev 
RUN apt-get install -y python3 python3-pip
RUN apt-get install -y \
    tzdata \
    libgflags-dev libhdf5-serial-dev libhdf5-dev \
    libsnappy-dev libatlas-base-dev
RUN apt-get install -y openssh-client openssh-server
RUN apt-get install -y openjdk-8-jdk

RUN echo "Asia/Shanghai" > /etc/timezone
ENV PATH /workspace/bm_prebuilt_toolchains/x86-64-core-i7--glibc--stable/bin:$PATH
ENV PATH /workspace/bm_prebuilt_toolchains/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin:$PATH
ENV PATH /workspace/bm_prebuilt_toolchains/mips-loongson-gcc7.3-linux-gnu/2019.06-29/bin:$PATH
ENV PATH /usr/sw/swgcc830_cross_tools-bb623bc9a/usr/bin:$PATH
ENV PATH /workspace/bm_prebuilt_toolchains/loongarch64-linux-gnu-2021-04-22-vector/bin:$PATH

WORKDIR /workspace
