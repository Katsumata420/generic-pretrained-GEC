MODEL=mass_ende_1024.pth

GPU=$1
seed=$2

CUDA_VISIBLE_DEVICES=$GPU python train.py \
  --exp_name gec_de_${seed}                            \
  --data_path ./data/processed/gec_de/                  \
  --lgs 'src-tgt'                                        \
  --seed $seed \
  --mt_steps 'src-tgt'                                 \
  --encoder_only false                                 \
  --emb_dim 1024                                       \
  --n_layers 6                                         \
  --n_heads 8                                          \
  --dropout 0.1                                        \
  --attention_dropout 0.1                              \
  --gelu_activation true                               \
  --tokens_per_batch 2000                              \
  --batch_size 32	                                     \
  --epoch_size -1                                       \
  --bptt 256                                           \
  --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
  --max_epoch 30                                       \
  --eval_bleu false                                    \
  --reload_model "$MODEL,$MODEL"                       \
