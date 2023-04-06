# HOW TO REPRODUCE

In our paper, we experiment with Czech, German, and Russian GEC as  the multilingual GEC.
This README explains how to reproduce the result, using Czech as an example.

## Environments
- AWS EC2 g5.xlarge
  - VRAM needs about 20 GB.
  - Deep Learning AMI GPU PyTorch 1.13.1 (Ubuntu 20.04)

## 1. Prepare
1. Install python (for m2scorer)
2. pytorch docker pull
3. git clone this repository
4. Get training data
5. Prepare tools
6. Prepare pretrained mBART model
7. Convert m2 data to src-tgt data
8. Tokenize with pretraiend spm

### 1.1. install python
```bash
$ sudo apt update && apt install -y python
```

### 1.1. pytorch docker pull
Use pytorch 1.4 with cuda 10.1 docker.

```bash
$ docker pull pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
```

### 1.2. git clone this repository
```bash
$ git clone https://github.com/Katsumata420/generic-pretrained-GEC.git
$ cd generic-pretrained-GEC/mBART-GEC
```

### 1.3. Get training data
Get Czech GEC data (AKCES-GEC.zip).

```bash
$ mkdir data && cd data  # /path/to/generic-pretrained-GEC/mBART-GEC/data
$ wget -O AKCES-GEC.zip https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3057/AKCES-GEC.zip?sequence=1&isAllowed=y
$ unzip AKCES-GEC.zip -d czech  # /path/to/generic-pretrained-GEC/mBART-GEC/data/czech
```

### 1.4. Prepare tools
Prepare the following tools:

- detokenizer in mosesdecoder
- tokenizer fo Czech (morphodita)
- sentencepiece
- python script
  - corr_from_m2.py: extract correct sentences from m2 file.
  - run_cs_tokenizer.py: tokenize Czech sentences using morphodita.
  - remove_unchanged_src_tgt.py: remove the unchanged sample between src and tgt.
  - m2scorer

```bash
$ cd /path/to/generic-pretrained-GEC/mBART-GEC
$ mkdir tools && cd tools  # /path/to/generic-pretrained-GEC/mBART-GEC/tools
# detokenizer in mosesdecoder
$ wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/detokenizer.perl
# tokenizer for Czech (morphodita)
$ wget https://github.com/ufal/morphodita/releases/download/v1.11.0/morphodita-1.11.0-bin.zip
$ unzip morphodita-1.11.0-bin.zip
# sentencepiece
$ git clone https://github.com/google/sentencepiece.git
$ cd sentencepiece
$ mkdir build
$ cmake ..
$ make
$ sudo make install
$ cd /path/to/generic-pretrained-GEC/mBART-GEC/tools
# corr_from_m2.py (https://gist.github.com/Katsumata420/7cf9c14cfcefc866d2c837ddd843c825)
$ git clone https://gist.github.com/7cf9c14cfcefc866d2c837ddd843c825.git
# run_cs_tokenizer.py (https://gist.github.com/Katsumata420/f17002cda766eeabf63437257ace3c76)
$ git clone https://gist.github.com/f17002cda766eeabf63437257ace3c76.git
# remove_unchanged_src_tgt.py (https://gist.github.com/Katsumata420/d1eb9d5d6fea729cee6c964c28218553)
$ git clone https://gist.github.com/d1eb9d5d6fea729cee6c964c28218553.git
# m2scorer
$ wget https://www.comp.nus.edu.sg/~nlp/sw/m2scorer.tar.gz
$ tar -xzvf m2scorer.tar.gz
```

### 1.5. Prepare pretraiend mBART model
```bash
$ cd /path/to/generic-pretrained-GEC/mBART-GEC
$ mkdir pretrain-models && cd pretrain-models  # /path/to/generic-pretrained-GEC/mBART-GEC/pretrain-models
$ wget https://dl.fbaipublicfiles.com/fairseq/models/mbart/mbart.CC25.tar.gz
$ tar -xzvf mbart.CC25.tar.gz
```

### 1.6. Convert m2 data to src-tgt data
Convert m2 train/valid/test files to parallel (src-tgt) file and preprocess its like a following:
- training data
  - convert m2 to parallel
  - remove unchanged src-tgt
  - remove duplicate sample
  - detokenize src-tgt
- validation data
  - convert m2 to parallel
  - detokenize src-tgt
- test data
  - convert m2 to parallel
  - detokenize src-tgt

```bash
$ cd /path/to/generic-pretrained-GEC/mBART-GEC/data/czech
# convert m2 to parallel
$ mkdir original
$ grep ^S train/train.all.m2 | cut -c 3- > original/train.src
$ grep ^S dev/dev.all.m2 | cut -c 3- > original/dev.src
$ grep ^S test/test.all.m2 | cut -c 3- > original/test.src
$ python3 ../../tools/7cf9c14cfcefc866d2c837ddd843c825/corr_m2_file.py train/train.all.m2 -out original/train.tgt
$ python3 ../../tools/7cf9c14cfcefc866d2c837ddd843c825/corr_m2_file.py dev/dev.all.m2 -out original/dev.tgt
$ python3 ../../tools/7cf9c14cfcefc866d2c837ddd843c825/corr_m2_file.py test/test.all.m2 -out original/test.tgt
# remove unchanged src-tgt (train)
$ mkdir unchanged
$ python3 ../../tools/d1eb9d5d6fea729cee6c964c28218553/remove_unchanged_src_tgt.py --src-file original/train.src --tgt-file original/train.tgt --output-dir unchanged
# remove duplicate sample (train)
$ mkdir deduplicate
$ paste unchanged/train.src.changed unchanged/train.tgt.changed | sort | uniq | shuf > deduplicate/train.src-tgt
$ cut -f1 deduplicate/train.src-tgt > deduplicate/train.src
$ cut -f2 deduplicate/train.src-tgt > deduplicate/train.tgt
# detokenize src-tgt
$ mkdir detokenize
$ perl ../../tools/detokenizer.perl -l cs < deduplicate/train.src > detokenize/train.src
$ perl ../../tools/detokenizer.perl -l cs < deduplicate/train.tgt > detokenize/train.tgt
$ perl ../../tools/detokenizer.perl -l cs < original/dev.src > detokenize/dev.src
$ perl ../../tools/detokenizer.perl -l cs < original/dev.tgt > detokenize/dev.tgt
$ perl ../../tools/detokenizer.perl -l cs < original/test.src > detokenize/test.src
$ perl ../../tools/detokenizer.perl -l cs < original/test.tgt > detokenize/test.tgt
```

### 1.7. Tokenize with pretrained spm
```bash
$ cd /path/to/generic-pretrained-GEC/mBART-GEC/data/czech
$ mkdir spm_data
$ spm_encode --model=../../pretrain-models/mbart.cc25/spm.model < detokenize/train.src > spm_data/train.spm.src
$ spm_encode --model=../../pretrain-models/mbart.cc25/spm.model < detokenize/train.tgt > spm_data/train.spm.tgt
$ spm_encode --model=../../pretrain-models/mbart.cc25/spm.model < detokenize/dev.src > spm_data/valid.spm.src  # dev -> valid
$ spm_encode --model=../../pretrain-models/mbart.cc25/spm.model < detokenize/dev.tgt > spm_data/valid.spm.tgt  # dev -> valid
$ spm_encode --model=../../pretrain-models/mbart.cc25/spm.model < detokenize/test.src > spm_data/test.spm.src  # only src for test
```

## 2. use fairseq
Before following command, modify the `preprocess_mbart.sh`, `fine_tune_mbart.sh`, and `translate_mbart.sh` script;

`preprocess_mbart.sh`: Set the DICT parameter.

`fine_tune_mbart.sh`: Set the PRETRAIN parameter.

`translate_mbart.sh`: Set the spm_model parameter.

1. Run docker container and go into it
2. pip install
3. Run preprocess
4. Run fine-tune
5. Translate test data using the fine-tuned model

### 2.1. Run docker container and go into it
```bash
$ cd /path/to/generic-pretrained-GEC/mBART-GEC
# prepare output dir
$ mkdir -p result/czech
$ docker run -it -v ./:/workspace/mBART --gpus all pytorch/pytorch:1.4-cuda10.1-cudnn7-devel bash
```

### 2.2. pip install
```bash
$ cd mBART && pwd
>> /workspace/mBART
$ pip install -e .
```

### 2.3. Run preprocess
```bash
# make fake data for test.tgt
$ cp data/czech/spm_data/test.spm.src data/czech/spm_data/test.spm.tgt
# run preprocess
$ bash preprocess_mbart.sh data/czech/spm_data czech_gec
```

### 2.4. Run fine-tune
```bash
$ bash fine_tune_mbart.sh 1 data-bin/czech_gec 0 cs_CZ
```

### 2.5. Translate test data using the fine-tuned model
```bash
$ bash translate_mbart.sh result/czech 0 model/mbart/gec_cs_CZ_1/checkpoint_best.pt data-bin/czech_gec cs_CZ
```


## 3. Evaluation 

1. Tokenize the generated text
2. Evaluate Result

### 3.1. Tokenize the generated text
```bash
# exit docker container (Ctrl-d)
$ cd /path/to/generic-pretrained-GEC/mBART-GEC
$ python3 tools/f17002cda766eeabf63437257ace3c76/run_cs_tokenizer.py \
  --input-file result/czech/hyp.cs_CZ.detok \
  --output-file result/czech/hyp.cs_CZ.tok \
  --bin-path tools/morphodita-1.11.0-bin/bin-linux64
```

### 3.2. Evaluate Result
```bash
$ cd /path/to/generic-pretrained-GEC/mBART-GEC
$ ./tools/m2scorer/m2scorer result/czech/hyp.cs_CZ.tok data/czech/test.all.m2
```