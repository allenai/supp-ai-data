# Data pipeline for supp.ai

The scripts in this repository are used to incrementally fetch data from S2 and extract supplement-drug interactions.

The pipeline consists of the following steps:

1. Download new data from S2 DB
2. Run NER/linking using scispaCy
3. Filter sentences for supp-drug or supp-supp entity pairs
4. Upload sentence candidates to Beaker
5. Run BERT-DDI classification model in Beaker
6. Download result datasets
7. Aggregate and reorganize evidence sentences
8. Upload final dictionaries to GCD

To run the whole pipeline, see `run_pipeline.sh`

This pipeline requires a configuration file, located at `config/config.json`, with the following format:

```json
{
  "timestamp": "2019-09-25T14:41:28.087503Z", 
  "rerun_ner": false, 
  "rerun_ddi": false
}
```

The timestamp indicates the last time the pipeline was run. Papers updated after the timestamp will be retreived and processed. If the timestamp is `None`, all PubMed papers from S2 will be retrieved.

The "rerun_ner" flag indicates whether NER should be re-run over all historical papers. If TRUE, BERT-DDI will also be re-run.

The "rerun_ddi" flag indicates whether the BERT-DDI model should be re-run over all historical papers.

The pipeline outputs the following log file:

```json
{
    "timestamp": "2019-09-25T14:41:28.087503Z", 
    "header_str": "20190925_01", 
    "raw_data_dir": "data/20190925_01/s2_data", 
    "entity_dir": "data/20190925_01/s2_entities", 
    "supp_sents_dir": "data/20190925_01/s2_supp_sents", 
    "ddi_output_dir": "data/20190925_01/ddi_output", 
    "aggregate": true, 
    "output_file": "output/20190925_07.tar.gz"
}
```
