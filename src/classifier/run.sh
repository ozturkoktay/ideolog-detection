#!/bin/bash

fuser -k 1246/tcp

#gunicorn --bind 0.0.0.0:1246 -k tornado -w 2 --threads 2 -t 2000 --log-level=DEBUG wsgi:app --daemon
gunicorn --bind 0.0.0.0:1246 -k tornado -w 2 --threads 2 -t 2000 --log-level=DEBUG wsgi:app
echo "Alive"
