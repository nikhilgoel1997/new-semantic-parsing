set -e
cd ..

DATE=Aug12
SET_NAME=top_iid
SPLIT_AMOUNTS=(0.99 0.95 0.90 0.85 0.7 0.5 0.3 0.1)

TAG="$SET_NAME"_"$DATE"_bert_run

for split_amount in "${SPLIT_AMOUNTS[@]}";
do

  data=data-bin/"$SET_NAME"_"$split_amount"_"$DATE"

  python cli/preprocess.py \
    --data data/top-dataset-semantic-parsing \
    --text-tokenizer bert-base-cased \
    --output-dir $data \
    --split-amount $split_amount \

done

# train

for split_amount in "${SPLIT_AMOUNTS[@]}";
do

  data=data-bin/"$SET_NAME"_"$split_amount"_"$DATE"
  model=models/"$SET_NAME"_"$split_amount"_"$DATE"_bert_run

  python cli/train_lightning.py \
    --data-dir $data  \
    --encoder-model bert-base-cased \
    --decoder-lr 0.2 \
    --encoder-lr 0.02 \
    --batch-size 192 \
    --layers 4 \
    --hidden 256 \
    --dropout 0.2 \
    --heads 4 \
    --label-smoothing 0.1 \
    --log-every 150 \
    --early-stopping 10 \
    --epochs 1000 \
    --max-steps 7000 \
    --min-steps 3000 \
    --warmup-steps 1500 \
    --freeze-encoder 0 \
    --unfreeze-encoder 500 \
    --output-dir $model \
    --tags train,"$TAG" \
    --split-amount-finetune $split_amount \


done


# finetune


for split_amount in "${SPLIT_AMOUNTS[@]}";
do

  data=data-bin/"$SET_NAME"_"$split_amount"_"$DATE"
  model=models/"$SET_NAME"_"$split_amount"_"$DATE"_bert_run


  for new_data_amount in 0.1 0.3 0.5 0.7 1.0
  do

    rm -rf output_dir/finetuned

      python cli/retrain.py \
        --data-dir $data \
        --model-dir $model \
        --batch-size 128 \
        --dropout 0.2 \
        --epochs 30 \
        --log-every 150 \
        --new-data-amount $new_data_amount \
        --tags finetune,$TAG \
        --split-amount-finetune $split_amount \
        --output-dir output_dir/finetuned \


  done

  for old_data_amount in 0.01 0.1
  do
  for new_data_amount in 0.1 1.0
  do

    rm -rf output_dir/finetuned

      python cli/retrain.py \
        --data-dir $data \
        --model-dir $model \
        --batch-size 128 \
        --dropout 0.2 \
        --epochs 30 \
        --log-every 150 \
        --new-data-amount $new_data_amount \
        --old-data-amount $old_data_amount \
        --tags finetune,$TAG \
        --split-amount-finetune $split_amount \
        --output-dir output_dir/finetuned \


  done
  done


done
