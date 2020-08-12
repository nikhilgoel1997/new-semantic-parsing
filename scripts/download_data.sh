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
