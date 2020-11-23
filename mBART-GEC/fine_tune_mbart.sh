seed=$1
BIN=$2
GPU=$3
language=$4

PRETRAIN=/path/to/cc25_pretrain/model.pt
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
out=model/mbart/gec_${language}_${seed}

TOTAL_NUM_UPDATES=20000
WARMUP_UPDATES=2000
LR=3e-05
MAX_TOKENS=1500
UPDATE_FREQ=2 

mkdir -p $out

CUDA_VISIBLE_DEVICES=$GPU python train.py $BIN  \
 --save-dir $out \
 --encoder-normalize-before \
 --decoder-normalize-before \
 --arch mbart_large \
 --task translation_from_pretrained_bart  \
 --source-lang src --target-lang tgt \
 --criterion label_smoothed_cross_entropy \
 --label-smoothing 0.1  --dataset-impl mmap \
 --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
 --lr-scheduler polynomial_decay --lr $LR \
 --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_NUM_UPDATES \
 --dropout 0.3 --attention-dropout 0.3  --weight-decay 0.01 \
 --max-tokens $MAX_TOKENS --update-freq $UPDATE_FREQ --save-interval 1 \
 --save-interval-updates 500 --keep-interval-updates 5 \
 --max-epoch 100 \
 --no-epoch-checkpoints \
 --seed $seed --log-format simple --log-interval 100 \
 --reset-optimizer --reset-meters --reset-dataloader \
 --reset-lr-scheduler --restore-file $PRETRAIN \
 --langs $langs --layernorm-embedding  --ddp-backend no_c10d \
 --gec_lang $language
