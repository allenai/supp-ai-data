#!/usr/bin/env bash

# activate conda environment
export PATH=$PATH:/home/lucyw/miniconda3/bin
cd /home/lucyw/git/supp-ai-data/
source activate suppai

# Get new papers and run through pipeline
echo 'Preprocessing data...'
python scripts/preprocess.py

echo 'Running beaker experiments...'
python scripts/run_beaker.py

echo 'Get S2 paper data...'
python scripts/get_pubmed_paper_info.py

echo 'Postprocessing data...'
python scripts/postprocess.py

# Upload to GCS and grant permissions
echo 'Uploading data to GCP...'
HEADER=$(jq -r '.header_str' config/log.json)
gsutil cp output/$HEADER.tar.gz -p ai2-reviz gs://supp-ai-data/
gsutil iam ch allUsers:roles/storage.legacyObjectReader gs://supp-ai-data/$HEADER.tar.gz

echo 'done.'