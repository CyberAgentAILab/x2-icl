#!/bin/bash -eu

MODEL="gpt-4o-2024-08-06"
SEED=1000

META="data/prompt/reasoning/nli_fewshot.txt"
TASK="ESNLI"
python code/make_all_reason.py --model_name ${MODEL} --reasoning_file ${META} --train_file data/prompt/full_labels/seed${SEED}/esnli_no.txt --test_file none --task ${TASK} --temp 0 --seed ${SEED}
python code/make_one_reason.py --train_file data/prompt/full_labels/seed${SEED}/esnli_all_reason.txt --test_file none

META="data/prompt/reasoning/nli_fewshot.txt"
TASK="ANLI"
python code/make_all_reason.py --model_name ${MODEL} --reasoning_file ${META} --train_file data/prompt/anli/seed${SEED}/anli_v1.txt --test_file none --task ${TASK} --temp 0 --seed ${SEED}
python code/make_one_reason.py --train_file data/prompt/anli/seed${SEED}/anli_v1_all_reason.txt --test_file none

META="data/prompt/reasoning/qqp_fewshot.txt"
TASK="QQP"
python code/make_all_reason.py --model_name ${MODEL} --reasoning_file ${META} --train_file data/prompt/full_labels/seed${SEED}/qqp_no.txt --test_file none --task ${TASK} --temp 0 --seed ${SEED}
python code/make_one_reason.py --train_file data/prompt/full_labels/seed${SEED}/qqp_all_reason.txt --test_file none
