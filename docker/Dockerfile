FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/base-cu122.py310
# Uninstall the container's TensorFlow version and install the latest version
RUN apt install libgomp1 libomp-dev -y

RUN pip install --upgrade pip && \
    pip install pycuda

RUN git clone https://github.com/gprMax/gprMax.git && \
    cd gprMax && \
    python setup.py build && \
    python setup.py install

RUN pip install gprmaxui --upgrade