
DICT=/path/to/dict.txt
DEST=data-bin
SRC=src
TGT=tgt
TRAIN=train
VALID=valid
TEST=test
DATA=$1
NAME=$2

python preprocess.py \
 --source-lang ${SRC} \
 --target-lang ${TGT} \
 --trainpref ${DATA}/${TRAIN}.spm \
 --validpref ${DATA}/${VALID}.spm \
 --testpref ${DATA}/${TEST}.spm \
 --destdir ${DEST}/${NAME} \
 --thresholdtgt 0 \
 --thresholdsrc 0 \
 --srcdict ${DICT} \
 --tgtdict ${DICT} \
 --workers 2
