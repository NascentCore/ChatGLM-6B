CUDA_VISIBLE_DEVICES='all'
FILE_DATE=$(date +%Y-%m-%d-%H)
TRAIN_EPOCHS=5
TRSIN_BATCH_SIZE=2

nohup deepspeed finetuning_freeze.py \
	--num_train_epochs $TRAIN_EPOCHS \
	--train_batch_size $TRSIN_BATCH_SIZE \
> ./output_log/freeze/result_ds_${FILE_DATE}_${TRAIN_EPOCHS}_${TRSIN_BATCH_SIZE}.out 2>&1 &

