#!/usr/bin/env bash

# 用法: ./data_collection_combined.sh [graspnet|contact] [起始编号]
# 示例: ./data_collection_combined.sh graspnet 1000

MODE=${1:-graspnet}       
BASE=${2:-0}             

GUI="False"
ITER=500
SCENE="pile"
GRASP_NUM=2000000
NUM=10
AVG_GRASP_NUM=$(( GRASP_NUM / NUM ))


if [ "$MODE" == "graspnet" ]; then
  echo "MODE: GRASPNET"
  for ((i=0; i<NUM; i++)); do
    START=$(( BASE + i * ITER ))
    echo "launch process $i : start_scene=$START"
    python generate_data_graspnet.py \
      --object-set "$SCENE/train" \
      --num-grasps "$AVG_GRASP_NUM" \
      --num-proc "1" \
      --save-scene \
      --root "data/data_pile_train_raw" &
  done

elif [ "$MODE" == "giga" ]; then
  echo "MODE: GIGA"
  for ((i=0; i<NUM; i++)); do
    START=$(( BASE + i * ITER ))
    echo "launch process $i : start_scene=$START"
    python generate_data_giga.py \
      --object-set "$SCENE/train" \
      --num-grasps "$AVG_GRASP_NUM" \
      --num-proc "1" \
      --save-scene \
      --root "data/data_pile_train_raw" &
  done

elif [ "$MODE" == "contact" ]; then
  echo "MODE: CONTACT"
  for ((i=0; i<NUM; i++)); do
    START=$(( BASE + i * ITER ))
    echo "launch process $i : start_scene=$START"
    python generate_data_contact.py \
      --scene "$SCENE" \
      --GUI "$GUI" \
      --start_scene "$START" \
      --iteration_num "$ITER" \
      --root "data/data_pile_train_raw" &
  done

else
  echo "错误：未知模式 '$MODE'"
  echo "用法: $0 [graspnet|contact] [起始编号]"
  exit 1
fi

# 等待所有后台任务完成
wait
echo "全部 $NUM 个进程已结束。"
