#!/bin/bash

# Deploy to Google Cloud Functions
# Note: ensure you have authenticated with `gcloud auth login` and set your project via `gcloud config set project <ID>`

FUNCTION_NAME=${FUNCTION_NAME:-predict-handler-v2}
PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-$PROJECT_ID}
REGION=${REGION:-us-central1}

if [ -z "$PROJECT_ID" ]; then
  echo "ERROR: PROJECT_ID (or GOOGLE_CLOUD_PROJECT) is not set." >&2
  exit 1
fi

gcloud functions deploy "$FUNCTION_NAME" \
  --project "$PROJECT_ID" \
  --runtime python311 \
  --trigger-http \
  --allow-unauthenticated \
  --memory 512MB \
  --timeout 3600s \
  --region "$REGION" \
  --source . \
  --entry-point predict_handler \
  --set-env-vars MODEL_SOURCE=${MODEL_SOURCE:-kaggle},MODEL_PATH=${MODEL_PATH:-model.pth},KAGGLE_DATASET=$KAGGLE_DATASET,KAGGLE_COMPETITION=$KAGGLE_COMPETITION,KAGGLE_NOTEBOOK=$KAGGLE_NOTEBOOK,HF_REPO_ID=$HF_REPO_ID,HF_MODEL_FILE=${HF_MODEL_FILE:-model.pth}

echo "Deployment completed. Check Google Cloud Console for function URL."
