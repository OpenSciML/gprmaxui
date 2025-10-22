FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/workbench-container:latest
# Uninstall the container's TensorFlow version and install the latest version
RUN apt install libgomp1 libomp-dev -y
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --upgrade pip && \
    pip install pycuda

RUN git clone https://github.com/gprMax/gprMax.git && \
    cd gprMax && \
    python setup.py build && \
    python setup.py install

RUN pip install gprmaxui --upgrade --force-reinstall --no-cache-dir