#!/bin/bash -eu
# NOTE: Ensure to run with `bash script/run_XX.sh`

MODEL="deepseek-r1-8b"
SEED=4000

ANLI_TESTS=('data/testset/anli_v1_hard.jsonl' 'data/testset/anli_v2_hard.jsonl' 'data/testset/anli_v3_hard.jsonl')
ANLI_TASKS=('ANLI' 'ANLI' 'ANLI')
ANLI_INSTS=('anli_v1.txt' 'anli_v1_one_reason.txt' 'anli_v1_all_reason.txt')
SNLI_TESTS=('data/testset/snli.json' 'data/testset/hans.json' 'data/testset/nan.json' 'data/testset/st.json' 'data/testset/pisp.json')
SNLI_TASKS=('ESNLI' 'HANS' 'NAN' 'ST' 'PISP')
SNLI_INSTS=('esnli_no.txt' 'esnli_one_reason.txt' 'esnli_all_reason.txt')
QQP_TESTS=('data/testset/qqp.json' 'data/testset/paws.json')
QQP_TASKS=('QQP' 'PAWS')
QQP_INSTS=('qqp_no.txt' 'qqp_one_reason.txt' 'qqp_all_reason.txt')

for i in ${!ANLI_TASKS[@]}; do \
    TASK=${ANLI_TASKS[i]}; \
    TEST=${ANLI_TESTS[i]}; \
    for INST in ${ANLI_INSTS[@]}; do \
        echo MODEL=$MODEL TASK=$TASK TEST=$TEST INST=$INST; \
        uv run code/run_hf.py \
            --model_name $MODEL \
            --train_file data/prompt/anli/seed$SEED/$INST \
            --test_file $TEST \
            --task $TASK \
            --sample 1 \
            --temp 0 \
            --seed $SEED \
            --dtype bf16 \
            ; \
    done; \
done

for i in ${!SNLI_TASKS[@]}; do \
    TASK=${SNLI_TASKS[i]}; \
    TEST=${SNLI_TESTS[i]}; \
    for INST in ${SNLI_INSTS[@]}; do \
        echo MODEL=$MODEL TASK=$TASK TEST=$TEST INST=$INST; \
        uv run code/run_hf.py \
            --model_name $MODEL \
            --train_file data/prompt/full_labels/seed$SEED/$INST \
            --test_file $TEST \
            --task $TASK \
            --sample 1 \
            --temp 0 \
            --seed $SEED \
            --dtype bf16 \
            ; \
    done; \
done

for i in ${!QQP_TASKS[@]}; do \
    TASK=${QQP_TASKS[i]}; \
    TEST=${QQP_TESTS[i]}; \
    for INST in ${QQP_INSTS[@]}; do \
        echo MODEL=$MODEL TASK=$TASK TEST=$TEST INST=$INST; \
        uv run code/run_hf.py \
            --model_name $MODEL \
            --train_file data/prompt/full_labels/seed$SEED/$INST \
            --test_file $TEST \
            --task $TASK \
            --sample 1 \
            --temp 0 \
            --seed $SEED \
            --dtype bf16 \
            ; \
    done; \
done

echo Done
