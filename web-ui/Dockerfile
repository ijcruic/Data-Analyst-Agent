FROM ubuntu:22.04

LABEL maintainer="SKOPE AI"
LABEL classification="UNCLASSIFIED"
LABEL description="Docker container for the Data Analyst Web UI"

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV XDG_CACHE_HOME=/app/xdg_cache

COPY sources.list /etc/apt/sources.list

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get dist-upgrade -y && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        gcc \
        ca-certificates \
    && \
    rm -rf /var/lib/apt/lists/*

COPY pip.conf /usr/pip.conf
COPY requirements.txt .
RUN rm -f /root/.config/pip/pip.conf /root/.pip/pip.conf /etc/pip.conf /etc/xdg/pip/pip.conf && \
    python3 -m pip install --upgrade pip wheel setuptools && \
    python3 -m pip install --no-cache-dir -r requirements.txt

COPY app.py config.py config.yaml README.md ./
RUN mkdir -p /tmp/uploads

EXPOSE 8000

CMD ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0", "--client.showErrorDetails=false"]
