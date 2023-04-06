# mBART-GEC
- Fine-tuned BART model for GEC.
  - for multilingual GEC.
- This script is based on fairseq.
  - original commit id: 08a1edd [[link]](https://github.com/pytorch/fairseq/tree/08a1edd667fb44f7587d305cea41c85800d004e2)
  - fairseq README: `FAIRSEQ_README.md`

## How To Run
[Translation example](https://github.com/Katsumata420/generic-pretrained-GEC/tree/master/mBART-GEC/examples/mbart) is based on GEC fine-tuning.

1. Prepare the GEC training data.
  - German: https://github.com/adrianeboyd/boyd-wnut2018#download-data
  - Czech: https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3057
  - Russian: https://github.com/arozovskaya/RULEC-GEC
2. Prepare the detokenizer and tokenizer for GEC language:
  - moses detokenizer: `wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/detokenizer.perl`
  - German tokenizer [[cite]](https://github.com/adrianeboyd/boyd-wnut2018#install-errant): `pip install -U spacy==2.0.11; python -m spacy download de`
  - Czech tokenizer: https://github.com/ufal/morphodita
  - Russian: https://github.com/aatimofeev/spacy_russian_tokenizer
3. Prepare pretrained mBART model (`mbart.CC25.tar.gz`).
```
wget https://dl.fbaipublicfiles.com/fairseq/models/mbart/mbart.CC25.tar.gz
tar -xzvf mbart.CC25.tar.gz
```
4. Detokenize GEC data using `detokenizer.perl`.
- `perl detokenizer.perl -l cs < tokenized_data > detokenized_data`
5. Install SentencePiece and use it for the detokenized data.
- SentencePiece: https://github.com/google/sentencepiece
```
SPM=/path/to/sentencepiece/build/src/spm_encode
DATA=/path/to/data
MODEL=sentence.bpe.model
TRAIN=train
VALID=valid
TEST=test
SRC=src
TGT=tgt
${SPM} --model=${MODEL} < ${DATA}/${TRAIN}.${SRC} > ${DATA}/${TRAIN}.spm.${SRC} &
${SPM} --model=${MODEL} < ${DATA}/${TRAIN}.${TGT} > ${DATA}/${TRAIN}.spm.${TGT} &
${SPM} --model=${MODEL} < ${DATA}/${VALID}.${SRC} > ${DATA}/${VALID}.spm.${SRC} &
${SPM} --model=${MODEL} < ${DATA}/${VALID}.${TGT} > ${DATA}/${VALID}.spm.${TGT} &
${SPM} --model=${MODEL} < ${DATA}/${TEST}.${SRC} > ${DATA}/${TEST}.spm.${SRC} &
${SPM} --model=${MODEL} < ${DATA}/${TEST}.${TGT} > ${DATA}/${TEST}.spm.${TGT} &
```
6. Binarize train/valid/test data.
```
# make fake data for test.tgt
mv /path/to/test.tgt /path/to/test.true.tgt
cp /path/to/test.src /path/to/test.tgt

# binarize
bash preprocess_mbart.sh /path/to/sp_data ${lang}_gec
```
7. Fine-tune the mBART model with binarized data.
- `bash fine_tune_mbart.sh 1 /path/to/data-bin $GPUID $lang `
8. Translate test data using the fine-tuned model.
- `bash translate_mbart.sh /path/to/result $GPUID /path/to/model /path/to/data-bin $lang`
9. Tokenized the generated text with the tokenizer.

## Fine-tuned Models
These fine-tuned models can be used to perform for each language.  
Use these models from step 8 onward in "How To Run".

### Czech
- fined-models (fine-tuned in April 2023.)
  - https://huggingface.co/Katsumata420/mBART-cz-GEC/blob/main/checkpoint_best.pt
- data-bin and test output
  - https://drive.google.com/drive/folders/1oECT9q06j9r0whKmp8cqgpzvXINFutoX?usp=share_link
  - m2score for this test output
    - pricision: 0.7575
    - recall: 0.6141
    - F0.5: 0.7237

### German
TBA

### Russian
TBA