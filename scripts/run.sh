set -e
cd ..


python cli/preprocess.py \
  --data data/top-dataset-semantic-parsing \
  --text-tokenizer bert-base-cased \
  --output-dir data-bin/top_jul13 \
  --split-class IN:GET_LOCATION \
  --split-amount 0.5 \


python cli/train_lightning.py \
  --data-dir data-bin/top_jul13 \
  --encoder-model bert-base-cased \
  --decoder-lr 0.2 \
  --encoder-lr 0.02 \
  --batch-size 128 \
  --gradient-accumulation-steps 2 \
  --layers 4 \
  --hidden 256 \
  --dropout 0.3 \
  --heads 4 \
  --label-smoothing 0.2 \
  --epochs 100 \
  --warmup-steps 1500 \
  --num-frozen-encoder-steps 500 \
  --log-every 500 \
  --early-stopping 10 \
  --output-dir output_dir/jul13_run \


for subsample in 0.1 0.3 0.5 0.7 0.9
do

    python cli/retrain.py \
      --data-dir data-bin/top_jul13 \
      --output-dir "output_dir/top_jul13_retrain_$subsample" \
      --model-dir output_dir/jul13_run \
      --lr 0.2 \
      --batch-size 128 \
      --gradient-accumulation-steps 2 \
      --dropout 0.3 \
      --label-smoothing 0.2 \
      --epochs 50 \
      --warmup-steps 500 \
      --num-frozen-encoder-steps 300 \
      --log-every 50 \
      --new-data-amount $subsample \
      --early-stopping 5 \


done

#python cli/predict.py \
#  --data data/top-dataset-semantic-parsing/test.tsv \
#  --model output_dir/jul10_run \
#  --output-file output_dir/jul10_run/test_predictions.tsv \
