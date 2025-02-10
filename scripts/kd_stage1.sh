
export ACCELERATE_USE_FSDP=1
export FSDP_CPU_RAM_EFFICIENT_LOADING=0
export NCCL_IB_DISABLE=1

CUDA_VISIBLE_DEVICES=0,1 nohup accelerate launch /data/hj/level4-cv-finalproject-hackathon-cv-18-lv3/train.py \
  --cfg-path /data/hj/level4-cv-finalproject-hackathon-cv-18-lv3/configs/train_distillation_stage1.yaml --kd > /data/hj/level4-cv-finalproject-hackathon-cv-18-lv3/train_log.out 2>&1 &
