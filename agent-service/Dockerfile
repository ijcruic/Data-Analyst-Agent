FROM ubuntu:22.04

LABEL maintainer="SKOPE AI"
LABEL classification="UNCLASSIFIED"
LABEL description="Docker container for a data analyst agent service"

WORKDIR /app

# Direct container to never require user input
ENV DEBIAN_FRONTEND=noninteractive
ENV XDG_CACHE_HOME=/app/xdg_cache

# Use production environment mirrors
COPY sources.list /etc/apt/sources.list

# Install system dependencies including Python toolchain
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get dist-upgrade -y && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        gcc \
        pandoc \
        ca-certificates \
    && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pip.conf /usr/pip.conf
COPY requirements.txt .
RUN rm -f /root/.config/pip/pip.conf /root/.pip/pip.conf /etc/pip.conf /etc/xdg/pip/pip.conf && \
    python3 -m pip install --upgrade pip wheel setuptools && \
    python3 -m pip install --no-cache-dir -r requirements.txt

# turn the readme into an HTML file to be served to users
COPY README.md .
RUN pandoc README.md -f gfm -t html -s -o README.html
RUN rm README.md

COPY agent.py main.py config.py config.yaml ./

EXPOSE 8002

CMD ["python3", "main.py"]
