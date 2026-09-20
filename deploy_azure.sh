#!/usr/bin/env bash
# Template deployment script for Azure Container Apps.
#
# Prerequisites:
#   1. Azure CLI installed (`az --version`) and logged in (`az login`).
#   2. An Azure subscription with permission to create resource groups / ACR / Container Apps.
#   3. GROQ_API_KEY exported in your shell: export GROQ_API_KEY="gsk_..."
#   4. Run this script from the project root (where the Dockerfile lives).
#
# This is a template. Claude cannot execute cloud deployments on your behalf — you must
# run this yourself after filling in/confirming the variables below.

set -euo pipefail

RESOURCE_GROUP="atliq-rag-rg"
LOCATION="eastus"
ACR_NAME="atliqragacr$RANDOM"          # must be globally unique; randomized suffix by default
APP_NAME="atliq-rag-chatbot"
ENV_NAME="atliq-rag-env"
IMAGE_NAME="rag-chatbot"
IMAGE_TAG="latest"

if [ -z "${GROQ_API_KEY:-}" ]; then
  echo "ERROR: export GROQ_API_KEY before running this script." >&2
  exit 1
fi

echo "Creating resource group..."
az group create --name "$RESOURCE_GROUP" --location "$LOCATION"

echo "Creating container registry..."
az acr create --resource-group "$RESOURCE_GROUP" --name "$ACR_NAME" --sku Basic

echo "Building and pushing image via ACR Tasks..."
az acr build --registry "$ACR_NAME" --image "$IMAGE_NAME:$IMAGE_TAG" .

echo "Creating Container Apps environment..."
az containerapp env create \
  --name "$ENV_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION"

echo "Deploying container app..."
az containerapp create \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --environment "$ENV_NAME" \
  --image "$ACR_NAME.azurecr.io/$IMAGE_NAME:$IMAGE_TAG" \
  --target-port 8501 \
  --ingress external \
  --registry-server "$ACR_NAME.azurecr.io" \
  --secrets groq-api-key="$GROQ_API_KEY" \
  --env-vars GROQ_API_KEY=secretref:groq-api-key GROQ_MODEL=llama-3.1-8b-instant \
  --cpu 1.0 --memory 2.0Gi

echo "Deployed. Public URL:"
az containerapp show --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" \
  --query properties.configuration.ingress.fqdn -o tsv

echo ""
echo "NOTE: chroma_store/ and logs/ are ephemeral inside the container by default."
echo "For production, mount Azure Files to those paths (az containerapp env storage set)"
echo "so the index and usage logs survive restarts/scale events."
