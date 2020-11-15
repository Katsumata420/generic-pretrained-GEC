OUT=$1
GPU=$2
model=$3
input=$4

mkdir -p $OUT

CUDA_VISIBLE_DEVICES=$GPU python translate_ems.py \
  $model \
  $input \
  $OUT
