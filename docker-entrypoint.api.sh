#!/usr/bin/env sh
# Entrypoint for Dockerfile.api. Builds the vector index at container
# STARTUP rather than image BUILD time (Phase 14 fix): the Kaggle dataset
# (data/olist.db, data/processed/rag_documents/) is gitignored and never
# baked into the image, so building the index at `docker build` time would
# only work on whoever's machine happens to have already run
# scripts/prepare_olist.py + scripts/generate_olist_rag_docs.py locally --
# it would fail in CI or on a clean clone. Instead: mount data/ and
# chroma_store/ as volumes (docker-compose.yml and deploy_azure.sh's Azure
# Files note both do this), and this script builds the index once, the
# first time the container starts against real data, then reuses it on
# every later start unless --rebuild-index is passed.
set -eu

if [ "${1:-}" = "--rebuild-index" ]; then
    echo "docker-entrypoint: --rebuild-index requested, rebuilding..."
    python ingest.py
    shift
elif [ ! -f "/app/chroma_store/chroma.sqlite3" ]; then
    echo "docker-entrypoint: no existing index found at /app/chroma_store, building one..."
    if [ ! -f "/app/data/olist.db" ]; then
        echo "docker-entrypoint: ERROR -- /app/data/olist.db not found. Mount your prepared" >&2
        echo "data/ directory as a volume (see docker-compose.yml / deploy_azure.sh's Azure" >&2
        echo "Files note) -- this image intentionally does not bundle the Kaggle dataset." >&2
        exit 1
    fi
    python ingest.py
else
    echo "docker-entrypoint: existing index found at /app/chroma_store, reusing it "
    echo "(pass --rebuild-index as the container command to force a rebuild)."
fi

exec "$@"
