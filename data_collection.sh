#!/usr/bin/env bash
# run.sh

# 可选：从命令行接收第一个参数作为起始 start_scene，默认为 0
BASE=${1:-0}

SCENE="pile/train"
GUI="False"
ITER=500
NUM=10
GRASP_NUM=200000

for ((i=0; i<NUM; i++)); do
  START=$(( BASE + i*ITER ))
  echo "启动进程 $i ： start_scene=$START"
  python generate_data_graspnet.py \
    --object-set "$SCENE" \
    --num-grasps "$GRASP_NUM" \
    --num-proc "0" \
    --save-scene \
    "data/data_pile_gn_train_raw" &
done

# 等待所有后台任务完成
wait
echo "全部 $NUM 个进程已结束。"