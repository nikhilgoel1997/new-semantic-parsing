# used to create a toy subset for test_zcli.py tests
# only uses train set, to avoid vocabulary issues

set -e
cd ..

mkdir data/top-dataset-semantic-parsing-1000
head -1000 data/top-dataset-semantic-parsing/train.tsv > data/top-dataset-semantic-parsing-1000/train.tsv
head -100 data/top-dataset-semantic-parsing/train.tsv > data/top-dataset-semantic-parsing-1000/eval.tsv
head -100 data/top-dataset-semantic-parsing/train.tsv > data/top-dataset-semantic-parsing-1000/test.tsv
