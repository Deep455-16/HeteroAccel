#!/bin/bash
git filter-branch -f --env-filter '
if [ "$GIT_AUTHOR_EMAIL" = "dev@adaptive-gpu-runtime.local" ]; then
    export GIT_AUTHOR_NAME="Deep Mundra"
    export GIT_AUTHOR_EMAIL="mundradeep17@gmail.com"
fi
if [ "$GIT_COMMITTER_EMAIL" = "dev@adaptive-gpu-runtime.local" ]; then
    export GIT_COMMITTER_NAME="Deep Mundra"
    export GIT_COMMITTER_EMAIL="mundradeep17@gmail.com"
fi
' --tag-name-filter cat -- --all
