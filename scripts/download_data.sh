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

mkdir data
cd data

curl -JLO http://fb.me/semanticparsingdialog
unzip semanticparsingdialog  # creates top-dataset-semantic-parsing folder
cd ..


git clone --depth=1 --branch=master https://github.com/snipsco/nlu-benchmark snips
rm -rf ./snips/.git

cd ../scripts

python process_snips.py --snips-path ../data/snips
