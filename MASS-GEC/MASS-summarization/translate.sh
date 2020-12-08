
MODEL=$1
GPU=$2
DATADIR=$3
OUT=$4

mkdir -p $OUT

CUDA_VISIBLE_DEVICES=$GPU fairseq-generate $DATADIR --path $MODEL \
    --user-dir mass --task translation_mass \
    --batch-size 32 --beam 4 \
    > $OUT/hyp.txt

cat $OUT/hyp.txt | grep "^H" | sort -V | cut -f 3- | sed 's/ ##//g' > $OUT/result.uncased.txt
