import torch
from fairseq.models.bart import BARTModel

import sys
import os

assert len(sys.argv) == 4, "translate_nbest.py input_text output_dir"
model_dir = 'model/gec_bart1'
input_text = sys.argv[1]
output_dir = sys.argv[2]
output_path = os.path.join(output_dir, "hyp.txt")


bart = BARTModel.from_pretrained(
    model_dir,
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='gec_data-bin'
)

bart2 = BARTModel.from_pretrained(
    'model/gec_bart2,
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='gec_data-bin'
)
bart3 = BARTModel.from_pretrained(
    'model/gec_bart3',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='gec_data-bin'
)
bart4 = BARTModel.from_pretrained(
    'model/gec_bart4',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='gec_data-bin'
)
bart5 = BARTModel.from_pretrained(
    'model/gec_bart5',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='gec_data-bin'
)

bart.cuda()
bart.eval()
bart.half()
bart2.cuda()
bart3.cuda()
bart4.cuda()
bart5.cuda()
bart2.eval()
bart3.eval()
bart4.eval()
bart5.eval()
bart2.half()
bart3.half()
bart4.half()
bart5.half()

count = 1
bsz = 16
beam_size = 12
output_count = 0

ensemble_models = [bart2.model, bart3.model, bart4.model, bart5.model]
# ensemble_models = []
with open(input_text) as source, open(output_path, 'w') as fout:
    sline = source.readline().strip()
    slines = [sline]
    for sline in source:
        if count % bsz == 0:
            print('current output count: {}'.format(output_count), file=sys.stderr)
            with torch.no_grad():
                hypotheses_batch = bart.sample_ensemble(slines, beam=beam_size, n_best=beam_size, ens_models=ensemble_models)

            for hypothesis in hypotheses_batch:
                sample_id, score, hypo_str = hypothesis.split('\t')
                sample_id = int(sample_id[2:])
                sample_id += bsz * output_count
                hypothesis = 'H-{}\t{}\t{}'.format(sample_id, score, hypo_str)
                fout.write(hypothesis + '\n')
                fout.flush()
            output_count += 1
            slines = []

        slines.append(sline.strip())
        count += 1
    if slines != []:
        hypotheses_batch = bart.sample_ensemble(slines, beam=beam_size, n_best=beam_size, ens_models=ensemble_models)
        for hypothesis in hypotheses_batch:
            sample_id, score, hypo_str = hypothesis.split('\t')
            sample_id = int(sample_id[2:])
            sample_id += bsz * output_count
            hypothesis = 'H-{}\t{}\t{}'.format(sample_id, score, hypo_str)
            fout.write(hypothesis + '\n')
            fout.flush()
print("fin")
