
FILE_NAME=$(date +%Y-%m-%d-%H) 

PRE_SEQ_LEN=128
LR=2e-2


#CUDA_VISIBLE_DEVICES=0
nohup python3 ../main.py \
    --do_train \
    --train_file ../../../data/pCLUE/history/pCLUE_train_1.csv \
    --prompt_column input \
    --response_column target \
    --overwrite_cache \
    --model_name_or_path ../../hub/models--THUDM--chatglm-6b/ \
    --output_dir ../output/adgen-clue-chatglm-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    > ../output_log/result_pclue_${FILE_NAME}_${PRE_SEQ_LEN}_${LR}.out 2>&1 &
