FROM chamelion:train-cu128
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libegl1 libxrender1 \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir polyscope==2.4.0
