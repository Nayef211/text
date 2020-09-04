#!/bin/bash
branch=$1
out_file=$2
vocab_benchmark_input_file=$3
is_jitable=$4
vocab_benchmark_file_dest_file=~/torchtext/vocab_benchmark.py

cp $vocab_benchmark_input_file $vocab_benchmark_file_dest_file
cd ~/torchtext
git checkout $branch

pip install -v -e .

echo "[Start Benchmarking] for $branch with is_jitable=$is_jitable"
# echo "[BRANCH] $branch" >> $out_file
python ~/torchtext/vocab_benchmark.py --is-jitable=$is_jitable >> $out_file
rm ~/torchtext/vocab_benchmark.py