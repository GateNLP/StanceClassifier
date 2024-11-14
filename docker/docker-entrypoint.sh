#!/bin/sh

# Allow for setup shell scripts to be injected to set up environment variables
# etc. before we execute the main process
for f in setup.d/*.sh ; do
  if [ -r "$f" ]; then
    . "$f"
  fi
done

exec /usr/bin/tini -- venv/bin/gunicorn -b=0.0.0.0:8000 "--workers=$WORKERS" "$@" elg_stance:app
