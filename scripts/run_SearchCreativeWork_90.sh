set -e
cd ..


# Train

SET_NAME=snips_search_creative_work_90
DATE=Aug12
CLASSES=IN:SEARCHCREATIVEWORK


DATA=data-bin/"$SET_NAME"_"$DATE"
MODEL=output_dir/"$SET_NAME"_"$DATE"_bert_run
TAG="$SET_NAME"_"$DATE"_bert_run


python cli/preprocess.py \
  --data data/snips/top_format \
  --text-tokenizer bert-base-cased \
  --output-dir $DATA \
  --split-class $CLASSES \
  --split-amount 0.90 \


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
  --tags train,$TAG \
  --new-classes $CLASSES \
  --seed 1 \


# Finetune

for new_data_amount in 0.1 0.3 0.5 0.7 1.0
do

    rm -rf output_dir/finetuned

    python cli/retrain.py \
      --data-dir $DATA \
      --model-dir $MODEL \
      --batch-size 128 \
      --dropout 0.2 \
      --epochs 40 \
      --log-every 100 \
      --new-data-amount $new_data_amount \
      --new-classes $CLASSES \
      --tags finetune,$TAG \
      --output-dir output_dir/finetuned \


done

for old_data_amount in 0.01 0.05 0.1 0.15 0.2 0.5 1.0
do

    rm -rf output_dir/finetuned

    python cli/retrain.py \
      --data-dir $DATA \
      --model-dir $MODEL \
      --batch-size 128 \
      --dropout 0.2 \
      --epochs 40 \
      --log-every 100 \
      --new-data-amount 1.0 \
      --old-data-amount $old_data_amount \
      --new-classes $CLASSES \
      --tags finetune,$TAG \
      --output-dir output_dir/finetuned \


done


for SEED in 2 3 4 5
do

  rm -rf "$MODEL"_seed

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
    --tags train,$TAG \
    --new-classes $CLASSES \
    --seed $SEED \

done
