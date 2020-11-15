TEST=$1
for LANG in src 
do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "$TEST.src" \
    --outputs "$TEST.bpe.$LANG" \
    --workers 10 \
    --keep-empty;
done
