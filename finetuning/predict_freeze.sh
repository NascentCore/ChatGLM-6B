CUDA_VISIBLE_DEVICES='all'
FILE_DATE=$(date +%Y-%m-%d-%H)

nohup python predict_freeze.py \
> ./output_log/predict/result_freeze_ds_${FILE_DATE}.out 2>&1 &
