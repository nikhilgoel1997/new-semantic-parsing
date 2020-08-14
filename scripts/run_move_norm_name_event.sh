set -e
cd ..

DATA=data-bin/top_name_event_95
MODEL=output_dir/name_event_95_bert_run
CLASSES=SL:NAME_EVENT
BATCH_SIZE=128

TAG=path_move_norm


for old_data_amount in 0.1 0.05 0.2
do

for reg in 0.1 0.0 1.0 0.01
do

    rm -rf output_dir/finetuned

    python cli/retrain.py \
      --data-dir $DATA \
      --model-dir $MODEL \
      --batch-size $BATCH_SIZE \
      --dropout 0 \
      --move-norm $reg \
      --label-smoothing 0.2 \
      --epochs 40 \
      --log-every 100 \
      --new-data-amount 1.0 \
      --old-data-amount $old_data_amount \
      --new-classes-file $CLASSES \
      --tags $TAG \
      --output-dir output_dir/finetuned \


    rm -rf output_dir/finetuned

    python cli/retrain.py \
      --data-dir $DATA \
      --model-dir $MODEL \
      --batch-size $BATCH_SIZE \
      --dropout 0 \
      --move-norm $reg \
      --label-smoothing 0.2 \
      --epochs 40 \
      --log-every 100 \
      --new-data-amount 1.0 \
      --old-data-amount $old_data_amount \
      --new-classes-file $CLASSES \
      --tags $TAG \
      --output-dir output_dir/finetuned \
      --move-norm-p 1 \


done
done
