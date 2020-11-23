
OUT=$1
GPU=$2
model=$3
BIN=$4
language=$5

langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
spm_model=/path/to/sentence.bpe.model
output_file=$OUT/output.${language}.txt

if [ $language = "cs_CZ" ]; then
    lang_token="\[cs_CZ\]"
elif [ $language = "de_DE" ]; then
    lang_token="\[de_DE\]"
elif [ $language = "ru_RU" ]; then
    lang_token="\[ru_RU\]"
else
    echo "set the language in cs_CZ, de_DE or ru_RU"
    exit 1
fi

mkdir -p $OUT

# `test.tgt` in $BIN is a copy data of `test.src`.
CUDA_VISIBLE_DEVICES=$GPU python generate.py $BIN  \
 --path $model  \
 --task translation_from_pretrained_bart \
 --gen-subset test -s src -t tgt --gec_lang $language \
 --bpe 'sentencepiece' --sentencepiece-vocab $spm_model \
 --remove-bpe 'sentencepiece' --max-sentences 32 --langs $langs \
 > $output_file

 cat $output_file | grep "^H" | sort -V | cut -f 3- | sed "s/${lang_token}//g" > $OUT/hyp.${language}.detok
