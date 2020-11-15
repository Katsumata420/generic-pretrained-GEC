for SPLIT in train valid
do
  for LANG in src tgt
  do
      python -m examples.roberta.multiprocessing_bpe_encoder \
          --encoder-json encoder.json \
          --vocab-bpe vocab.bpe \
          --inputs "/path/to/$SPLIT.$LANG" \
          --outputs "/path/to/$SPLIT.bpe.$LANG" \
          --workers 10 \
          --keep-empty;
    done
done

