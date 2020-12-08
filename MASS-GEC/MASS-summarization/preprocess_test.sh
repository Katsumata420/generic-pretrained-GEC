fairseq-preprocess \
    --user-dir mass --task masked_s2s \
    --source-lang src --target-lang tgt \
    --testpref data/gec_en/jfleg.bpe \
    --destdir data-bin/gec_en_jfleg --srcdict model/pretrained/dict.txt --tgtdict model/pretrained/dict.txt \
    --workers 10
