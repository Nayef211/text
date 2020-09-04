import glob
import json
import os
import subprocess
import datetime

VOCAB_BRANCHES = ["nayef211/vocab_py_dict_pybind", "nayef211/vocab_c10dict_pybind", "nayef211/vocab_fast_text_pybind",
                "nayef211/vocab_fast_text_dict", "nayef211/vocab_python_dict_efficient_list_lookup",
                "nayef211/vocab_order_preserving_flat_hashmap"]


def repro_all_vocab_benchmarks(vocab_benchmark_file, out_file):
    # print current date time
    with open(out_file, "a+") as f:
        now = datetime.datetime.now()
        f.write("\n----------" + now.strftime("%Y-%m-%d %H:%M:%S") + "----------")

    for branch in VOCAB_BRANCHES:
        with open(out_file, "a+") as f:
            f.write("\n[BRANCH] " + branch + "\n")

        # pybind registered vocab is not JITable 
        is_jitable = "pybind" not in branch

        print("Running benchmark for ", branch)
        subprocess.check_call(['./reproduce_benchmarks.sh', branch, out_file, vocab_benchmark_file, str(is_jitable)])


def main():
    vocab_benchmark_file = os.path.join(os.getcwd(), "vocab_benchmark.py")
    vocab_results_file = os.path.join(os.getcwd(), "vocab_results.txt")
    repro_all_vocab_benchmarks(vocab_benchmark_file=vocab_benchmark_file, out_file=vocab_results_file)

    print("Path to Vocab benchmark results: {}".format(vocab_results_file))


if __name__ == "__main__":
    main()