#!/bin/bash
# cats all the source file in the correct order (and removes lines containing the anchor as comment)
# so that the output can be copy-pasted into a kaggle notebook directly and executed


ANCHOR='ANCHOR_REMOVE_LINE_KAGGLE'

usage() {
  echo "usage: $0 <train|eval>"
  exit 1 
}

cat_src_file() {
    file=$1
    grep -v $ANCHOR $file    
    echo
}

common() {
    cat_src_file src/util.py
    cat_src_file src/data.py
    cat_src_file src/model/init_weights.py
    cat_src_file src/model/unet3_submodules.py
    cat_src_file src/model/unet3.py
}

train() {
    common
    cat_src_file src/train.py
}

evaluate() {
    common
    cat_src_file src/eval.py
}

if [[ "$#" != 1 ]]; then
  usage
fi

case "$1" in
  "train")
    train
    ;;
  "eval")
    evaluate
    ;;
  *) usage ;;
esac
