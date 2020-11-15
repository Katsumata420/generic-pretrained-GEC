fairseq-preprocess \
  --source-lang "src" \
  --target-lang "tgt" \
  --trainpref "/path/to/train.bpe" \
  --validpref "/path/to/valid.bpe" \
  --destdir "gec_data-bin/" \
  --workers 10 \
  --srcdict dict.txt \
  --tgtdict dict.txt;
