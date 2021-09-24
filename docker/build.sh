#!/bin/bash

if [ -z "$TAG" ]; then
  TAG=elg/stance_$1:latest
fi

exec docker buildx build -f Dockerfile -t $TAG --build-arg LANGUAGE=$1 ..
