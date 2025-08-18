set -ex

PROMPT_TYPE=$1
MODEL_NAME_OR_PATH=$2
MAX_TOKEN=$3
NUM_SHOTS=$4
DATASETS=$5
OUTPUT_DIR=$6

SPLIT="test"
NUM_TEST_SAMPLE=-1

# English open datasets
DATA_NAME=${DATASETS}
TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --max_tokens_per_call ${MAX_TOKEN} \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --num_shots ${NUM_SHOTS} \
    --save_outputs \
    --overwrite \

# get response statistics
python data_process.py ${MODEL_NAME_OR_PATH} "outputs/"${OUTPUT_DIR}
