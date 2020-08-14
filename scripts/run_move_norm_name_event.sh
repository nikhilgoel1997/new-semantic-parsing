# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

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
