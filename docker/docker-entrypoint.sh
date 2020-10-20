#!/bin/sh

ls -alh

exec /sbin/tini -- venv/bin/hypercorn --bind=0.0.0.0:8000 "--workers=$WORKERS" "$@" elg_stance:app
