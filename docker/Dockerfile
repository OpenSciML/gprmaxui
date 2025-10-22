FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/workbench-container:latest

# Install required system dependencies
RUN apt-get update && \
    apt-get install -y \
        libgomp1 \
        libomp-dev \
        ffmpeg \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install PyCUDA
RUN pip install --upgrade pip && \
    pip install pycuda cython

# Clone gprMax repository and install Python dependencies before building
RUN git clone https://github.com/gprMax/gprMax.git && \
    cd gprMax && \
    python setup.py build && \
    python setup.py install

# Install gprMax UI (fresh install)
RUN pip install --upgrade --force-reinstall --no-cache-dir gprmaxui
