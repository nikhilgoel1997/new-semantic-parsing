# Incremental Retraining for Semantic Parsing

## Introduction

In this work we study how to efficiently add new data to a semantic parsing model without retraining it from scratch.
Experiments include finetuning on new data, finetuning with subsampling from the old data and regularization techniques
that improve the final preformance of the model and/or reduce the need of using large amounts of old data.

### Installation

To work with the repository you need to install required packages and this package.
Edit (-e) mode is perferred if you want to change the code.

```bash
pip install -r requirements.txt
pip install -e .
```

### Usage

`scripts/download_data.sh` downloads TOP and SNIPS datasets.
It also reformats SNIPS into TOP format.

```bash
# download data

sh scripts/download_data.sh
```

Preprocess script splits train set into pretrain and finetune parts, creates tokenizers, numericalizes the data and saves in to `--output-dir` folder.

```bash

# preprocess

DATA=data-bin/top_dataset

python cli/preprocess.py \
  --data data/top-dataset-semantic-parsing \
  --text-tokenizer bert-base-cased \
  --split-amount 0.9 \
  --output-dir $DATA \
```

Train script trains the model on the pretrain part and saves the model and the trainer to `--output-dir` folder.
We recommend the following hyperparameters for training.

```bash
# train

DATA=data-bin/top_dataset
MODEL=output_dir/top_model

python cli/train_lightning.py \
  --data-dir $DATA  \
  --encoder-model bert-base-cased \
  --decoder-lr 0.2 \
  --encoder-lr 0.02 \
  --batch-size 192 \
  --layers 4 \
  --hidden 256 \
  --dropout 0.2 \
  --heads 4 \
  --label-smoothing 0.1 \
  --epochs 100 \
  --warmup-steps 1500 \
  --freeze-encoder 0 \
  --unfreeze-encoder 500 \
  --log-every 150 \
  --early-stopping 10 \
  --output-dir $MODEL \
```

Retrain script loads the model and optimizer from the checkpoint and finetunes on the finetune part of the training set.

```bash
DATA=data-bin/top_dataset
MODEL=output_dir/top_model

python cli/retrain.py \
  --data-dir $DATA \
  --model-dir $MODEL \
  --batch-size 128 \
  --dropout 0.2 \
  --epochs 40 \
  --log-every 100 \
  --old-data-amount 0.1 \
  --move-norm 0.1 \
```

## Run scripts

You can find more usage examples in the `scripts` directory.

## Disclaimer
This is not an officially supported Google product
