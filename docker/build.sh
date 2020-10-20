#!/bin/bash

TAG=$1

if [ -z "$TAG" ]; then
  TAG=elg/stance:latest
fi

exec docker build -f Dockerfile -t $TAG ..
