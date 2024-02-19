export GLUE_DIR=$HOME/glue_data
export TASK_NAME=RTE

CUDA_VISIBLE_DEVICES=4 python3 run_glue.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 64 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ./$TASK_NAME/ \
  --overwrite_output_dir \
  --local_rank 0 \
  --master_ip localhost \
  --master_port 6585 \
  --world_size 2
