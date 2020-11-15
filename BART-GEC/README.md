# BART-GEC
- Fine-tuned BART model for GEC.
  - Only English GEC.
- This script is based on fairseq.
  - original commit id: 9f4256e [[link]](https://github.com/pytorch/fairseq/tree/9f4256edf60554afbcaadfa114525978c141f2bd)
  - fairseq README: `FAIRSEQ_README.md`

## How To Run
[Summarization example](https://github.com/Katsumata420/generic-pretrained-GEC/blob/master/BART-GEC/examples/bart/README.cnn.md) is used for GEC fine-tuning.

1. Prepare the BEA-train/valid/test data (Lang-8, NUCLE, and so on).
    - https://www.cl.cam.ac.uk/research/nl/bea2019st/#data
2. Prepare pretrained BART model (`bart.large.tar.gz`) and related files
 (`encoder.json`, `vocab.bpe` and `dict.txt`).
    - `bart.large.tar.gz`: [url](https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz)
    - `encoder.json`: [url](https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json)
    - `vocab.bpe`: [url](https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe)
    - `dict.txt`: [url](https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt)
2. Apply BART BPE to the BEA-train/valid data.
    - Use `apply_bpe.sh`.
3. Binarize train/valid data.
    - Use `preprocess.sh`.
4. Fine-tune the BART model with binarized data.
    - Use `train.sh`.
5. Translate BEA-test using the fine-tuned model.
    - Use `translate.sh`.

### Caution
Lewis et al. have instructed to run BPE on non-tokenization text in CNN/DailyMail.

However, I didn't detokenized the text.

I used the original tokenization in BEA-train and applied BPE to the tokenized BEA-train text.

