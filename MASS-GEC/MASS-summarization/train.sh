SEED=$1
GPU=$2

OUT=model/mbart/gec_en_$SEED
mkdir -p $OUT
batch=3000
warm_up=4000

CUDA_VISIBLE_DEVICES=$GPU fairseq-train data-bin/gec_en \
    --seed $SEED \
    --save-dir $OUT --log-format simple --log-interval 100 \
    --source-lang src --target-lang tgt \
    --user-dir mass --task translation_mass --arch transformer_mass_middle \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.0005 --min-lr 1e-09 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $warm_up \
    --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --update-freq 8 --max-tokens $batch \
    --ddp-backend=no_c10d --max-epoch 10 \
    --max-source-positions 512 --max-target-positions 512 \
    --skip-invalid-size-inputs-valid-test \
    --load-from-pretrained-model model/pretrained/mass-middle-uncased.pt \
