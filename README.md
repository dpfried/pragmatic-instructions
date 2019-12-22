# Unified Pragmatic Models for Generating and Following Instructions
Daniel Fried, Jacob Andreas, and Dan Klein

[NAACL, 2018](https://arxiv.org/abs/1711.04987)

This repository currently contains the code for follower and speaker models for the SCONE domains. SAIL is not currently integrated; please contact Daniel Fried if you'd like the code for it, or for other questions about the code.

## Requirements

### Packages
- python 3.6
- dynet 2.0
- numpy
- scipy
- pandas

### Data
- The [SCONE dataset](https://nlp.stanford.edu/projects/scone/). Place the `rlong` directory inside `data/scone`.

## Training

Train a base follower (aka listener, in the paper) or speaker model with 

`train_follower.sh <corpus> <seed>` or `train_speaker.sh <corpus> <seed>`

where `<corpus>` is one of `alchemy`, `tangrams`, or `scene`, and `<seed>` is an integer. We used ensembles of models with seeds 1-10 for the results in the paper. Hyperparameters used in the paper (which achieved the best results on the dev set for each corpus) are hardcoded into these scripts.

Models and results will be output to `expts/{follower,speaker}/<corpus>/<model_name>/<seed>`.

## Evaluation

These scripts will evaluate ensembles of the models, in both the literal setting (using only scores from the ensemble of models used to produce candidates) and the rational setting (using scores from both speakers and listeners). Results will be output to `expts/rational_{speaker,follower}/<corpus>/<model>`.

`eval_followers.sh <corpus>` and `eval_speakers.sh <corpus>`

By default, these scripts will look for 10 follower and speaker models trained with random seeds 1 through 10 (see Training above).
