

LR=1e-4
FILE_NAME=$(date +%Y-%m-%d-%H) 

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

nohup deepspeed --num_gpus=2 --master_port $MASTER_PORT main_wandb.py \
    --deepspeed deepspeed.json \
    --do_train \
    --train_file AdvertiseGen/train.json \
    --test_file AdvertiseGen/dev.json \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path THUDM/chatglm-6b \
    --output_dir ./output/adgen-chatglm-6b-ft-$LR-wandb \
    --overwrite_output_dir \
    --max_source_length 32 \
    --max_target_length 32 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --fp16 True \
> ./output_log/result_${LR}_ds_${FILE_NAME}.out 2>&1 &


