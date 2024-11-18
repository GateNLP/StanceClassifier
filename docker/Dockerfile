FROM python:3.12-slim as base

# Install tini and create an unprivileged user
RUN apt-get update && \
    apt-get install -y \
      tini \
    && rm -rf /var/lib/apt/lists/* && \
    addgroup --gid 1001 "elg" && \
    adduser --disabled-password --gecos "ELG User,,," \
      --home /elg --ingroup elg --uid 1001 elg && \
    install -m 700 -g 1001 -o 1001 -d /elg/.cache

# Copy in requirements file for the venv
COPY requirements.txt /elg/

# Everything from here down runs as the unprivileged user account
USER 1001:1001

WORKDIR /elg

# Create a Python virtual environment for the dependencies
# We mount a cache which we use for pip. This means that
# repeat builds are much much quicker as we don't need to
# re-download all the dependencies. In this instance we've
# used a none default directory so that ~/.cache doesn't end
# up owned by root and so that when we run the classifier
# torch can cache it's downloads in the normal place
RUN --mount=type=cache,target=/elg/.pip-cache,uid=1001 python -mvenv venv && venv/bin/pip --cache-dir /elg/.pip-cache install -r requirements.txt

COPY --chown=1001:1001 StanceClassifier /elg/StanceClassifier
COPY --chown=1001:1001 docker/elg_stance.py docker/docker_classifier.py docker/docker-entrypoint.sh /elg/

# copy in any extra models from the context, which can subsequently be referenced as
# ./extra-models/<name> in the build args.  The [s] is a trick so that this instruction
# won't fail if (as in the default case) no such extra models exist.
COPY --chown=1001:1001 docker/extra-model[s] /elg/extra-models

ENTRYPOINT ["./docker-entrypoint.sh"]

FROM base as oblivious

ARG MODEL=default
ENV STANCE_CLASSIFIER_MODE=oblivious STANCE_CLASSIFIER_OBLIVIOUS_MODEL=$MODEL

# Run one classify, to pre-cache any download-on-first-use things
RUN ["venv/bin/python", "-c", "from docker_classifier import classifier\nclassifier.classify({'text':'reply'})"]

ENV WORKERS=1 REQUEST_SIZE_LIMIT=100000 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

FROM base as aware

ARG MODEL=default
ENV STANCE_CLASSIFIER_MODE=aware STANCE_CLASSIFIER_AWARE_MODEL=$MODEL

RUN ["venv/bin/python", "-c", "from docker_classifier import classifier\nclassifier.classify_with_target({'text':'reply', 'id_str':'2', 'in_reply_to_status_id_str':'1'}, {'text':'orig', 'id_str':'1'})"]

ENV WORKERS=1 REQUEST_SIZE_LIMIT=100000 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

FROM base as ensemble

ARG MODEL_TO=default
ARG MODEL_TA=default
ENV STANCE_CLASSIFIER_MODE=ensemble STANCE_CLASSIFIER_OBLIVIOUS_MODEL=$MODEL_TO STANCE_CLASSIFIER_AWARE_MODEL=$MODEL_TA

RUN ["venv/bin/python", "-c", "from docker_classifier import classifier\nclassifier.classify_with_target({'text':'reply', 'id_str':'2', 'in_reply_to_status_id_str':'1'}, {'text':'orig', 'id_str':'1'})"]

ENV WORKERS=1 REQUEST_SIZE_LIMIT=100000 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
