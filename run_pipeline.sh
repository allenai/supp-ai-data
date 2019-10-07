#!/usr/bin/env bash

# activate conda environment
source activate suppai

# Get new papers and run through pipeline
echo 'Preprocessing data...'
python scripts/preprocess.py

echo 'Running beaker experiments...'
python scripts/run_beaker.py

echo 'Postprocessing data...'
python scripts/postprocess.py

# Upload to GCS and grant permissions
echo 'Uploading data to GCP...'
HEADER=$(jq -r '.header_str' config/log.json)
gsutil cp output/$HEADER.tar.gz -p ai2-reviz gs://supp-ai-data/
gsutil iam ch allUsers:roles/storage.legacyObjectReader gs://supp-ai-data/$HEADER.tar.gz

echo 'done.'