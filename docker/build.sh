#!/bin/bash

if [ -z "$TAG" ]; then
  TAG=elg/stance_english_reply_only:latest
fi

exec docker buildx build -f Dockerfile -t $TAG --build-arg MODEL=$1 ..
