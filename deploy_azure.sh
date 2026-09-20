#!/usr/bin/env bash
# Deployment script for Azure Container Apps -- Phase 15, two-service
# architecture (Phase 12/13 split the app into an API backend and a
# Streamlit UI; this deploys both as separate Container Apps sharing one
# ACR + environment, matching docker-compose.yml's local topology).
#
# Prerequisites:
#   1. Azure CLI installed (`az --version`) and logged in (`az login`).
#   2. An Azure subscription with permission to create resource groups / ACR / Container Apps.
#   3. GROQ_API_KEY exported in your shell: export GROQ_API_KEY="gsk_..."
#   4. JWT_SECRET_KEY exported: export JWT_SECRET_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
#      (do NOT reuse your local .env's JWT_SECRET_KEY for a real deployment without
#      thinking about it -- generating a fresh one here is intentional and fine).
#   5. Run this script from the project root (where Dockerfile.api / Dockerfile.ui live).
#   6. Your prepared data/ directory (data/olist.db, data/processed/rag_documents/) --
#      see README.md's Quickstart -- since the image no longer bakes it in (Phase 14
#      fix: it's gitignored and built into the index at container startup instead).
#
# This is a real, runnable script -- Claude cannot execute cloud deployments on your
# behalf, so you must run it yourself after filling in/confirming the variables below.

set -euo pipefail

RESOURCE_GROUP="rbac-rag-agent-rg"
LOCATION="eastus"
ACR_NAME="rbacragagentacr$RANDOM"      # must be globally unique; randomized suffix by default
API_APP_NAME="rbac-rag-agent-api"
UI_APP_NAME="rbac-rag-agent-ui"
ENV_NAME="rbac-rag-agent-env"
API_IMAGE_NAME="rag-api"
UI_IMAGE_NAME="rag-ui"
IMAGE_TAG="latest"
GROQ_MODEL="${GROQ_MODEL:-openai/gpt-oss-20b}"   # matches config.py's own default; override if your .env uses a different model

if [ -z "${GROQ_API_KEY:-}" ]; then
  echo "ERROR: export GROQ_API_KEY before running this script." >&2
  exit 1
fi
if [ -z "${JWT_SECRET_KEY:-}" ]; then
  echo "ERROR: export JWT_SECRET_KEY before running this script (see prerequisite 4 above)." >&2
  exit 1
fi
if [ ! -f "data/olist.db" ]; then
  echo "ERROR: data/olist.db not found. This script uploads your local data/ directory to" >&2
  echo "Azure Files below (prerequisite 6) -- prepare it first per README.md's Quickstart." >&2
  exit 1
fi

echo "Creating resource group..."
az group create --name "$RESOURCE_GROUP" --location "$LOCATION"

echo "Creating container registry..."
az acr create --resource-group "$RESOURCE_GROUP" --name "$ACR_NAME" --sku Basic

echo "Building and pushing both images via ACR Tasks..."
az acr build --registry "$ACR_NAME" --image "$API_IMAGE_NAME:$IMAGE_TAG" -f Dockerfile.api .
az acr build --registry "$ACR_NAME" --image "$UI_IMAGE_NAME:$IMAGE_TAG" -f Dockerfile.ui .

echo "Creating Container Apps environment..."
az containerapp env create \
  --name "$ENV_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION"

echo "Creating Azure Files storage for the vector index, usage logs, and Kaggle dataset..."
STORAGE_ACCOUNT="rbacragagentst$RANDOM"
az storage account create --name "$STORAGE_ACCOUNT" --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" --sku Standard_LRS
STORAGE_KEY=$(az storage account keys list --resource-group "$RESOURCE_GROUP" \
  --account-name "$STORAGE_ACCOUNT" --query "[0].value" -o tsv)
for share in chroma-store logs olist-data; do
  az storage share-rm create --resource-group "$RESOURCE_GROUP" \
    --storage-account "$STORAGE_ACCOUNT" --name "$share"
done
az containerapp env storage set \
  --name "$ENV_NAME" --resource-group "$RESOURCE_GROUP" \
  --storage-name chroma-store-link --azure-file-account-name "$STORAGE_ACCOUNT" \
  --azure-file-account-key "$STORAGE_KEY" --azure-file-share-name chroma-store \
  --access-mode ReadWrite
az containerapp env storage set \
  --name "$ENV_NAME" --resource-group "$RESOURCE_GROUP" \
  --storage-name logs-link --azure-file-account-name "$STORAGE_ACCOUNT" \
  --azure-file-account-key "$STORAGE_KEY" --azure-file-share-name logs \
  --access-mode ReadWrite
az containerapp env storage set \
  --name "$ENV_NAME" --resource-group "$RESOURCE_GROUP" \
  --storage-name olist-data-link --azure-file-account-name "$STORAGE_ACCOUNT" \
  --azure-file-account-key "$STORAGE_KEY" --azure-file-share-name olist-data \
  --access-mode ReadOnly

echo "Uploading your local data/ directory to Azure Files (one-time; re-run if the dataset changes)..."
az storage file upload-batch --account-name "$STORAGE_ACCOUNT" --account-key "$STORAGE_KEY" \
  --destination olist-data --source data

echo "Deploying API container app..."
az containerapp create \
  --name "$API_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --environment "$ENV_NAME" \
  --image "$ACR_NAME.azurecr.io/$API_IMAGE_NAME:$IMAGE_TAG" \
  --target-port 8000 \
  --ingress internal \
  --registry-server "$ACR_NAME.azurecr.io" \
  --secrets groq-api-key="$GROQ_API_KEY" jwt-secret-key="$JWT_SECRET_KEY" \
  --env-vars GROQ_API_KEY=secretref:groq-api-key GROQ_MODEL="$GROQ_MODEL" \
             JWT_SECRET_KEY=secretref:jwt-secret-key \
             API_CORS_ORIGINS="https://${UI_APP_NAME}.internal.${LOCATION}.azurecontainerapps.io" \
  --cpu 1.0 --memory 2.0Gi

API_FQDN=$(az containerapp show --name "$API_APP_NAME" --resource-group "$RESOURCE_GROUP" \
  --query properties.configuration.ingress.fqdn -o tsv)

echo "Deploying UI container app (public-facing, talks to the API over the internal network)..."
az containerapp create \
  --name "$UI_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --environment "$ENV_NAME" \
  --image "$ACR_NAME.azurecr.io/$UI_IMAGE_NAME:$IMAGE_TAG" \
  --target-port 8501 \
  --ingress external \
  --registry-server "$ACR_NAME.azurecr.io" \
  --env-vars API_BASE_URL="https://$API_FQDN" \
  --cpu 0.5 --memory 1.0Gi

echo ""
echo "Deployed. Public UI URL:"
az containerapp show --name "$UI_APP_NAME" --resource-group "$RESOURCE_GROUP" \
  --query properties.configuration.ingress.fqdn -o tsv

echo ""
echo "NOTE: the --secrets/--env-vars calls above set the API's runtime config, but the"
echo "Azure Files mounts (chroma-store-link, logs-link, olist-data-link) created above"
echo "still need to be attached as volumes on the API container app via"
echo "'az containerapp update --name $API_APP_NAME ... --yaml' with a volumes/volumeMounts"
echo "block -- the az CLI does not yet expose per-container volume mounts as flat flags"
echo "the way it does env-vars. See:"
echo "https://learn.microsoft.com/en-us/azure/container-apps/storage-mounts for the YAML shape."
echo "Without this last step the API container starts with EMPTY chroma_store/data/logs"
echo "directories and will rebuild the index from nothing (and fail, since data/olist.db"
echo "won't be there either) -- do not skip it."
