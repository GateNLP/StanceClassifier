FROM python:3.7-slim

# Install tini and create an unprivileged user
ADD https://github.com/krallin/tini/releases/download/v0.19.0/tini /sbin/tini
RUN addgroup --gid 1001 "elg" && \
      adduser --disabled-password --gecos "ELG User,,," \
      --home /elg --ingroup elg --uid 1001 elg && \
      chmod +x /sbin/tini

# Copy in requirements file for the venv
COPY requirements.txt /elg/

# Everything from here down runs as the unprivileged user account
USER elg:elg

WORKDIR /elg

# Create a Python virtual environment for the dependencies
# We mount a cache which we use for pip. This means that
# repeat builds are much much quicker as we don't need to
# re-download all the dependencies. In this instance we've
# used a none default directory so that ~/.cache doesn't end
# up owned by root and so that when we run the classifier
# torch can cache it's downloads in the normal place
RUN --mount=type=cache,target=/elg/.pip-cache,uid=1001 python -mvenv venv && venv/bin/pip --cache-dir /elg/.pip-cache install -r requirements.txt

# Resources are shared across all models
COPY resources /elg/resources

# Copy in the relevant model
ARG LANGUAGE
COPY models/BERT_$LANGUAGE /elg/models/cloud_docker

COPY StanceClassifier /elg/StanceClassifier
COPY docker/elg_stance.py docker/docker-entrypoint.sh resources.txt /elg/

# Run one classify, to pre-cache any download-on-first-use things
RUN ["venv/bin/python", "-c", "from StanceClassifier.stance_classifier import StanceClassifier\nStanceClassifier('cloud-docker').classify({'text':'original'},{'text':'reply'})"]

ENV WORKERS=1 REQUEST_SIZE_LIMIT=100000

ENTRYPOINT ["./docker-entrypoint.sh"]
