FROM nvcr.io/nvidia/pytorch:22.11-py3

RUN rm -rf /workspace/*
WORKDIR /workspace

# COPY --chown=$USER:$USER . .
ADD requirements.txt .
ADD . .
RUN pip install --no-cache-dir --upgrade --pre pip
RUN pip install --no-cache-dir -r requirements.txt

