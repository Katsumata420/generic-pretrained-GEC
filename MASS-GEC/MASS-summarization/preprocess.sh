fairseq-preprocess \
    --user-dir mass --task masked_s2s \
    --source-lang src --target-lang tgt \
    --trainpref data/gec_en/train.bpe --validpref data/gec_en/valid.bpe --testpref data/gec_en/test.bpe \
    --destdir data-bin/gec_en --srcdict model/pretrained/dict.txt --tgtdict model/pretrained/dict.txt \
    --workers 10
