OUT=$1
GPU=$2
model=$3

mkdir -p $OUT

CUDA_VISIBLE_DEVICES=$GPU python translate.py \
  $model \
  /path/to/ABCN.test.bea19.orig \
  $OUT
