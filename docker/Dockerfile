FROM ubuntu:16.04
COPY pip.conf /etc/pip.conf
# Ali apt-get source.list
RUN mv /etc/apt/sources.list /etc/apt/sources.list.bak && \
    echo "deb-src http://archive.ubuntu.com/ubuntu xenial main restricted" >/etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/ubuntu/ xenial main restricted" >>/etc/apt/sources.list && \
    echo "deb-src http://mirrors.aliyun.com/ubuntu/ xenial main restricted multiverse universe" >>/etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/ubuntu/ xenial-updates main restricted" >>/etc/apt/sources.list && \
    echo "deb-src http://mirrors.aliyun.com/ubuntu/ xenial-updates main restricted multiverse universe" >>/etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/ubuntu/ xenial universe" >>/etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/ubuntu/ xenial-updates universe" >>/etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/ubuntu/ xenial multiverse" >>/etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/ubuntu/ xenial-updates multiverse" >>/etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/ubuntu/ xenial-backports main restricted universe multiverse" >>/etc/apt/sources.list && \
    echo "deb-src http://mirrors.aliyun.com/ubuntu/ xenial-backports main restricted universe multiverse" >>/etc/apt/sources.list && \
    echo "deb http://archive.canonical.com/ubuntu xenial partner" >>/etc/apt/sources.list && \
    echo "deb-src http://archive.canonical.com/ubuntu xenial partner" >>/etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/ubuntu/ xenial-security main restricted" >>/etc/apt/sources.list && \
    echo "deb-src http://mirrors.aliyun.com/ubuntu/ xenial-security main restricted multiverse universe" >>/etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/ubuntu/ xenial-security universe" >>/etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/ubuntu/ xenial-security multiverse" >>/etc/apt/sources.list &&\
apt-get update && \
    apt-get install -y \
        cmake \
        libtool \
	wget \
        git \
	vim \
        tree \
        python-dev python3-dev python-paramiko \
        python-pip python3-pip python-cffi python-numpy python-scipy \
        libgfortran3 libopenblas-dev \
        libleveldb-dev \
        libssl-dev liblmdb-dev dialog \
        pkg-config libopencv-dev libncurses5-dev \
        libgflags-dev libhdf5-10 libhdf5-serial-dev libhdf5-dev \
        libsnappy-dev libatlas-base-dev && \
git clone -b 4.1.0 --depth=1 https://github.com/opencv/opencv.git && \
    cd opencv && \
    mkdir build && \
    cd build && \
    cmake -DWITH_LIBV4L=ON .. && \
    make -j$(nproc) && \
    make install && \
    cd ../../ && rm -rf opencv && \
mkdir -p /data/release/toolchains/gcc && \
    cd /data/release/toolchains/gcc && \
    wget https://releases.linaro.org/components/toolchain/binaries/6.3-2017.05/aarch64-linux-gnu/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu.tar.xz && \
    tar xf gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu.tar.xz && \
    rm -f gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu.tar.xz && \
mkdir -p /tmp/googletest && \
    cd /tmp/googletest && \
    git clone https://github.com/google/googletest && \
    cd googletest && \
    mkdir build && cd build && \
    cmake .. && make && make install && \
    cd /tmp && rm -rf /tmp/googletest && \
    pip install --upgrade pip && \
    pip3 install --upgrade pip &&\
ldconfig && \
    apt-get clean && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install numpy==1.14.6\
    sphinx sphinx-autobuild sphinx_rtd_theme recommonmark\
    lmdb \
    backports-abc==0.5 \
    blinker==1.4 \
    bokeh==1.0.4 \
    certifi==2019.3.9 \
    chardet==3.0.4 \
    Click==7.0 \
    decorator==4.0.6 \
    dominate==2.3.5 \
    Flask==1.0.2 \
    Flask-Bootstrap==3.3.7.1 \
    Flask-DebugToolbar==0.10.1 \
    Flask-Session==0.3.1 \
    Flask-SQLAlchemy==2.3.2 \
    funcsigs==1.0.2 \
    idna==2.8 \
    ipython \
    itsdangerous==1.1.0 \
    Jinja2==2.10 \
    MarkupSafe==1.1.1 \
    mock==2.0.0 \
    numexpr==2.6.9 \
    packaging==19.0 \
    pandas==0.24.2 \
    pbr==5.1.3 \
    pexpect==4.0.1 \
    Pillow==5.4.1 \
    ptyprocess==0.5 \
    pyparsing==2.3.1 \
    python-dateutil==2.8.0 \
    pytz==2018.9 \
    PyYAML==5.1 \
    requests==2.21.0 \
    simplegeneric==0.8.1 \
    singledispatch==3.4.0.3 \
    six==1.12.0 \
    SQLAlchemy==1.3.1 \
    subprocess32==3.5.3 \
    tables==3.5.1 \
    tornado==5.1.1 \
    tqdm==4.31.1 \
    urllib3==1.24.1 \
    visitor==0.1.3 \
    Werkzeug==0.15.0 \
 &&  pip3 install tensorflow==1.13.1\
         numpy==1.14.6 \
         torchvision==0.2.1\
         torch==1.0.0\
         graphviz==0.8.4\
         mxnet==1.4.1\
         gluoncv opencv-contrib-python PyYAML \
         sphinx sphinx-autobuild sphinx_rtd_theme recommonmark \
lmdb parso  jedi  wcwidth  prompt-toolkit  decorator  ptyprocess  pexpect  pickleshare  ipython-genutils  traitlets  backcall  ipython  Cython  nose  opencv-python  networkx  python-gflags  retrying  plotly  leveldb  jupyter-core  pyzmq  jupyter-client  ipykernel  attrs  pyrsistent  jsonschema  nbformat  Send2Trash  prometheus-client  terminado  testpath  entrypoints  mistune  defusedxml  pandocfilters  webencodings  bleach  nbconvert  notebook  widgetsnbextension  ipywidgets  qtconsole  jupyter-console  jupyter  PyWavelets  imageio  scikit-image  pandas && \
rm -rf ~/.cache/pip/*

ENV PATH /data/release/toolchains/gcc/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin:$PATH

WORKDIR /workspace

