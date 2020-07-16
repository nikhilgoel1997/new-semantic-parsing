set -e
cd ..

DATA=data-bin/top_path01_jul15
MODEL=output_dir/get_path01_jul15_bert_run
CLASSES=data/splits/path_related.tsv
TAG=path1_bert


python cli/preprocess.py \
  --data data/top-dataset-semantic-parsing \
  --text-tokenizer bert-base-cased \
  --output-dir $DATA \
  --split-class SL:PATH \
  --split-amount 0.99 \


python cli/train_lightning.py \
  --data-dir $DATA  \
  --encoder-model bert-base-cased \
  --decoder-lr 0.2 \
  --encoder-lr 0.02 \
  --batch-size 192 \
  --layers 4 \
  --hidden 256 \
  --dropout 0.3 \
  --heads 4 \
  --label-smoothing 0.1 \
  --epochs 100 \
  --warmup-steps 1500 \
  --num-frozen-encoder-steps 500 \
  --log-every 150 \
  --early-stopping 10 \
  --output-dir $MODEL \
  --new-classes-file $CLASSES \
  --tag train \


for subsample in 0.1 0.3 0.5 1.0
do

    python cli/retrain.py \
      --data-dir $DATA \
      --model-dir $MODEL \
      --batch-size 192 \
      --dropout 0.1 \
      --label-smoothing 0.2 \
      --epochs 50 \
      --log-every 5 \
      --new-data-amount $subsample \
      --new-classes-file $CLASSES \
      --early-stopping 10 \
      --tag $TAG \


done
