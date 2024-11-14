#!/bin/bash

TARGET=$1
shift

if [ -z "$TAG" ]; then
  TAG=stance-classifier:latest
fi

exec docker buildx build -f Dockerfile -t $TAG --target=$TARGET "$@" ..
