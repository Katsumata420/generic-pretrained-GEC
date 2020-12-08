input=$1 # bpe sentence
model=$2
output=$3
gpu=$4

beam=4
bsz=32

cat $input | \
     CUDA_VISIBLE_DEVICES=$gpu python translate.py --exp_name translate \
     --beam $beam --batch_size $bsz \
     --src_lang de --tgt_lang de \
     --model_path $model --output_path $output

