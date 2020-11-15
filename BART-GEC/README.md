# BART-GEC
- Fine-tuned BART model for GEC.
  - English GEC
- This script is based on fairseq.
  - original commit id: 9f4256e [[link]](https://github.com/pytorch/fairseq/tree/9f4256edf60554afbcaadfa114525978c141f2bd)
  - fairseq README: FAIRSEQ_README.md

## How To Run
[Summarization example]() is used for GEC fine-tuning.

1. Prepare the BEA-train/valid/test data (Lang-8, NUCLE, and so on).
  - https://www.cl.cam.ac.uk/research/nl/bea2019st/#data
2. Prepare pretrained BART model (bart.large.tar.gz) and related files (encoder.json, vocab.bpe and dict.txt).
  - path
2. Apply BART BPE to the BEA-train/valid data.
3. Binarize train/valid data.
4. Fine-tune the BART model with binarized data.
5. Translate BEA-test using the fine-tuned model.

### Caution
Lewis et al. have instructed to run BPE on non-tokenization text in CNN/DailyMail.
However, I didn't detokenized the text.
I used the original tokenization in BEA-train and applied BPE to the tokenized BEA-train text.

