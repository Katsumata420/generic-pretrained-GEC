# MASS-SUM
original README: `MASS_README.md`
<!---
The source sentence in summarization tasks is usually long. To handle the long sequence, we use document-level corpus to extract long-continuous sequence for pre-training. The max sequence length is set as 512 for each iteration. For each sequence, we randomly mask a continuous segment for every 64-tokens at the encoder and predict it at the decoder. 
-->

## Dependency
```
pip install torch==1.0.0 
pip install fairseq==0.8.0
```

## MODEL
MASS uses default Transformer structure. We denote L, H, A as the number of layers, the hidden size and the number of attention heads. 

| Model | Encoder | Decoder | Download |
| :------| :-----|:-----|:-----|
| MASS-base-uncased | 6L-768H-12A | 6L-768H-12A | [MODEL](https://modelrelease.blob.core.windows.net/mass/mass-base-uncased.tar.gz) | 
| MASS-middle-uncased | 6L-1024H-16A | 6L-1024H-16A | [MODEL](https://modelrelease.blob.core.windows.net/mass/mass-middle-uncased.tar.gz) |

## How To Run
- WIP

