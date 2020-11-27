FROM python:3.7-slim

ARG LANGUAGE

# Install tini and create an unprivileged user
ADD https://github.com/krallin/tini/releases/download/v0.19.0/tini /sbin/tini
RUN addgroup --gid 1001 "elg" && \
      adduser --disabled-password --gecos "ELG User,,," \
      --home /elg --ingroup elg --uid 1001 elg && \
      chmod +x /sbin/tini

# Copy in our app, its requirements file and the entrypoint script
COPY requirements.txt /elg/

# Everything from here down runs as the unprivileged user account
USER elg:elg

WORKDIR /elg

# Create a Python virtual environment for the dependencies
RUN python -mvenv venv && venv/bin/pip --no-cache-dir install -r requirements.txt

COPY resources /elg/resources
COPY models/BERT_$LANGUAGE /elg/models/cloud_docker

COPY StanceClassifier /elg/StanceClassifier
COPY docker/elg_stance.py docker/docker-entrypoint.sh resources.txt /elg/

ENV WORKERS=1 REQUEST_SIZE_LIMIT=100000

ENTRYPOINT ["./docker-entrypoint.sh"]
