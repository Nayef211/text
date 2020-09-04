#!/bin/bash
out_file=pipeline_benchmark_results.txt
benchmark_path=~/torchtext/examples/data_pipeline/pipelines.py

vocab_filename=/private/home/nayef211/dev/dataset_vocabs/ag_news_vocab.txt
dataset=AG_NEWS

# vocab_filename=/private/home/nayef211/dev/dataset_vocabs/amazon_review_full_vocab.txt
# dataset=AmazonReviewFull

--vocab-filename=$vocab_filename --dataset=$dataset


echo "[Start Pipeline Benchmarks]"
echo "---------- `date` ----------" >> $out_file
echo "[DATASET] {$dataset}"
echo "[VOCAB FILENAME] {$vocab_filename}"

numactl --membind 0 --cpubind 0 taskset -c 0 python $benchmark_path --pipeline legacy_torchtext --vocab-filename=$vocab_filename --dataset=$dataset >> $out_file
numactl --membind 0 --cpubind 0 taskset -c 0 python $benchmark_path --pipeline experimental_torchtext --vocab-filename=$vocab_filename --dataset=$dataset >> $out_file

numactl --membind 0 --cpubind 0 taskset -c 0 python $benchmark_path --pipeline legacy_pytext_vocab --vocab-filename=$vocab_filename --dataset=$dataset >> $out_file
numactl --membind 0 --cpubind 0 taskset -c 0 python $benchmark_path --pipeline legacy_pytext_script_vocab --vocab-filename=$vocab_filename --dataset=$dataset >> $out_file
numactl --membind 0 --cpubind 0 taskset -c 0 python $benchmark_path --pipeline experimental_pytext_script_vocab --vocab-filename=$vocab_filename --dataset=$dataset >> $out_file

numactl --membind 0 --cpubind 0 taskset -c 0 python $benchmark_path --pipeline legacy_fasttext --vocab-filename=$vocab_filename --dataset=$dataset >> $out_file
numactl --membind 0 --cpubind 0 taskset -c 0 python $benchmark_path --pipeline experimental_fasttext --vocab-filename=$vocab_filename --dataset=$dataset >> $out_file
