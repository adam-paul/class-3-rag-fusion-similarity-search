#!/bin/bash

if [ "$1" = "main" ]; then
    docker compose run --rm rag_app
elif [ "$1" = "upload" ]; then
    docker compose run --rm upload_service
elif [ "$1" = "jupyter" ]; then
    docker compose up jupyter
elif [ "$1" = "chat_repl" ]; then
    docker compose run --rm chat_repl
elif [ "$1" = "rag_slack_repl" ]; then
    docker compose run --rm rag_slack_repl
elif [ "$1" = "seed_messages" ]; then
    docker compose run --rm seed_messages
else
    docker compose run --rm rag_app python "$1"
fi
