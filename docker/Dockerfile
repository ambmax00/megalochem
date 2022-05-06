FROM ubuntu:20.04
# basically evrything from Dockerfile from dbcsr

ENV DEBIAN_FRONTEND=noninteractive

# install basics
RUN set -ex ; \
    apt-get update ; \
    apt-get install -y \
        apt-transport-https \
        ca-certificates \
        gnupg \
        software-properties-common \
        curl \
        ; \
    curl -LsS https://apt.kitware.com/keys/kitware-archive-latest.asc \
        | gpg --dearmor - \
        | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null ; \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main' ; \
    apt-get install -y \
        locales \
        gfortran \
        g++ \
        openmpi-bin \
        libopenmpi-dev \
        mpich \
        libmpich-dev \
        libopenblas-openmp-dev \
        cmake \
        ninja-build \
        lcov \
        pkg-config \
        git \
        python3-pip \
        unzip \
        rsync \
        valgrind \
        ; \
    rm -rf /var/lib/apt/lists/* ; \
    localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8

# Make sure we have a proper unicode locale and language
ENV LANG en_US.utf8

RUN set -ex ; \
    pip3 install \
        pre-commit \
        ford \
        git-archive-all \
        ;

# install libxsmm
ARG libxsmm_version=1.17

RUN set -ex ; \
    curl -LsS https://github.com/hfp/libxsmm/archive/${libxsmm_version}.tar.gz | tar -xz -C /opt ; \
    ln -s libxsmm-${libxsmm_version} /opt/libxsmm ; \
    make -j -C /opt/libxsmm MALLOC=0

ENV PKG_CONFIG_PATH="/opt/libxsmm/lib:${PKG_CONFIG_PATH}"

# install OneAPI HPC kit
# repository to install Intel(R) oneAPI Libraries
RUN curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB | apt-key add -
RUN echo "deb [trusted=yes] https://apt.repos.intel.com/oneapi all main " > /etc/apt/sources.list.d/oneAPI.list
RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    curl ca-certificates gpg-agent software-properties-common && \
  rm -rf /var/lib/apt/lists/*
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  ca-certificates build-essential pkg-config gnupg libarchive13 \
  intel-basekit-getting-started intel-oneapi-common-licensing intel-oneapi-common-vars \
  intel-oneapi-mkl-devel intel-hpckit-getting-started && \
  rm -rf /var/lib/apt/lists/*

# install eigen
WORKDIR /tmp/eigen
RUN curl -o eigen3.tar.gz https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
RUN gunzip < eigen3.tar.gz | tar xf - 
WORKDIR eigen-3.4.0/build
RUN cmake -DCMAKE_INSTALL_PREFIX=/opt/eigen3 .. \
    && make -j install
WORKDIR /

# install hdf5
WORKDIR /tmp/hdf5
RUN curl -o hdf5.tar.gz https://hdf-wordpress-1.s3.amazonaws.com/wp-content/uploads/manual/HDF5/HDF5_1_12_2/source/hdf5-1.12.2.tar.gz
RUN gunzip < hdf5.tar.gz | tar xf -
WORKDIR hdf5-1.12.2/build
RUN cmake -DHDF5_ENABLE_PARALLEL=ON \
    -DBUILD_TESTING:BOOL=OFF \
    -DCMAKE_INSTALL_PREFIX=/opt/hdf5 \
    .. \
    && make -j install
WORKDIR /

# install libcint
WORKDIR /tmp/libcint
RUN git clone https://github.com/sunqm/libcint.git libcint 
WORKDIR libcint/build 
RUN cmake -DWITH_COULOMB_ERF=on \
    -DWITH_RANGE_COULOMB=on \
    -DENABLE_STATIC=on \
    -DCMAKE_INSTALL_PREFIX=/opt/libcint \
    .. \
    && make -j install 
WORKDIR /

# install dbcsr
WORKDIR /tmp/dbcsr
RUN git clone --recursive https://github.com/cp2k/dbcsr.git dbcsr
WORKDIR dbcsr/build 
RUN pkg-config libxsmm libxsmmext --libs \
    && cmake \
        -DCMAKE_INSTALL_PREFIX=/opt/dbcsr \
        -DUSE_SMM=libxsmm \
        .. \
    && make -j install
WORKDIR /

ENV MKLROOT=/opt/intel/oneapi/mkl/latest/