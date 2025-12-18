FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# system deps (git useful for submodules; build tools for some pip packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# copy only requirements first for better caching
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# copy the rest of your code
COPY . /workspace

# default command (you can override it)
CMD ["bash"]
