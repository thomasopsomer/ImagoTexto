FROM phusion/baseimage:latest

RUN apt-get -y update \
  && apt-get -y upgrade \
  && apt-get -y install git-core build-essential gcc g++ gfortran \
  && apt-get -y install python-pip wget libgfortran3 imagemagick cmake\
  && apt-get install -y --no-install-recommends software-properties-common


# install openblas
RUN cd /tmp && \
	git clone https://github.com/xianyi/OpenBLAS.git && \
	cd OpenBLAS && \
	make NO_AFFINITY=1 USE_OPENMP=1 && \
	make install 

RUN echo "/opt/OpenBLAS/lib" > /etc/ld.so.conf.d/openblas.conf \
	&& ldconfig

# install python lib
# clone numpy, scipy
RUN cd /tmp && \
	git clone git://github.com/numpy/numpy.git && \
	git clone -q git://github.com/scipy/scipy.git

# Add config
ADD numpy-site.cfg /tmp/numpy/site.cfg
ADD scipy-site.cfg /tmp/scipy/site.cfg

# install numpy, scipy against OpenBLAS
RUN pip install cython && \
	cd /tmp/numpy && python setup.py install && \
	cd /tmp/scipy && python setup.py install && \
	cd /root

# install overfeat
RUN git clone https://github.com/sermanet/OverFeat && \
	cd OverFeat && ./download_weights.py && \
	cd src && make all

# install overfeat python bindings
RUN ln -s /opt/OpenBLAS/lib/libopenblas.so /usr/lib/libopenblas.so && \
	cd /OverFeat/API/python && python setup.py install && \
	pip install pillow

ENV OVERFEAT_HOME /OverFeat