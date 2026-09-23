FROM python:3.10-slim-bookworm@sha256:2559be987fd64d61badbdafd303ea58a9ccab36d6c3c08bce219e762177d2eca
ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ git libeigen3-dev libgomp1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install numpy==1.26.4 open3d==0.18.0 PyYAML==6.0.2 \
    matplotlib==3.9.4 pytest==8.3.5 scikit-build-core==0.11.6 pybind11==2.13.6 cmake==3.31.6
# Override upstream's floating pybind11 master fetch without changing TRAVEL.
RUN git clone --depth 1 --branch v2.13.6 https://github.com/pybind/pybind11.git /opt/pybind11
COPY third_party/TRAVEL /opt/TRAVEL
COPY third_party/TRAVEL_REVISION /opt/TRAVEL_REVISION
RUN python -m pip install ninja==1.11.1.4
RUN CMAKE_ARGS="-DFETCHCONTENT_SOURCE_DIR_PYBIND11=/opt/pybind11" \
    python -m pip install --no-build-isolation --no-deps /opt/TRAVEL/python
WORKDIR /workspace

# Interactive generation dependencies and CPU OpenGL/X11 support.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-dri libegl1 libxrender1 libsm6 libxext6 xvfb xauth \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install trimesh==4.6.6 filterpy==1.4.5 opencv-python==4.11.0.86
