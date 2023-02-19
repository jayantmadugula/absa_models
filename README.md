# Aspect-Based Sentiment Analysis (ABSA) Models

This repository includes a few different ABSA models. It is still being updated to support more datasets, models, and experiments.

To train a model, use the `train_absa_model_script.py`.

For models that require metadata, sentiment labels can be generated using the `infer_metadata_script.py`.

Pre-saved embeddings can be generated using the `generate_embeddings_script.py` file.

All scripts have help pages (use the `-h` flag) with information on required and optional arguments.

## Parameters

The `parameters.json` file contains required parameters for the above scripts. 
Note: please create directories for script outputs before running the scripts.