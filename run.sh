#!/bin/bash

if [ "$1" = "main" ]; then
    docker compose run --rm rag_app
elif [ "$1" = "upload" ]; then
    docker compose run --rm upload_service
elif [ "$1" = "jupyter" ]; then
    docker compose up jupyter
else
    docker compose run --rm rag_app python "$1"
fi
