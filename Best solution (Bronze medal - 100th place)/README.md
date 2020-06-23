## Quick summary
- Model: 5-Fold `XLM_RoBERTa large` using the last layer `CLS` token.
  - 5-Fold `XLM_RoBERTa large` concatenated `AVG` and `MAX` last layer pooled with 8 multi-sample dropout.
- Inference: I have used `TTA` for each sample, I predicted on `head`, `tail`, and a mix of both tokens.
- Labels: `toxic` cast to `int` then pseudo-labels on the test set.
- Dataset: For the 1st model I used a 1:2 ratio between toxic and not toxic samples, for the 2nd model use 1:1 ratio, also did some basic cleaning. (details below)
- Framework: `Tensorflow` with TPU.

## Detailed summary

### Model &amp; training
- 1st model [link for the 1st fold training](https://github.com/dimitreOliveira/Jigsaw-Multilingual-Toxic-Comment-Classification/blob/master/Model%20backlog/Train/99-jigsaw-fold1-xlm-roberta-large-best.ipynb)
  - 5-Fold XLM_RoBERTa large with the last layer `CLS` token
  - Sequence length: 192
  - Batch size: 128
  - Epochs: 4
  - Learning rate: 1e-5
  - Training schedule: Exponential decay with 1 epoch warm-up
  - Losses: BinaryCrossentropy on the labels cast to `int`

Trained the model 4 epochs on the train set then 1 epoch on the validation set, after that 2 epochs on the test set with pseudo-labels.

- 2nd model [link for the 1st fold training](https://github.com/dimitreOliveira/Jigsaw-Multilingual-Toxic-Comment-Classification/blob/master/Model%20backlog/Train/136-jigsaw-fold1-xlm-roberta-ratio-1-8-sample-drop.ipynb)
  - 5-Fold XLM_RoBERTa large with the last layer `AVG` and`MAX` pooled concatenated then fed to 8 multi-sample dropout
  - Sequence length: 192
  - Batch size: 128
  - Epochs: 3
  - Learning rate: 1e-5
  - Training schedule: Exponential decay with 10% of steps warm-up
  - Losses: BinaryCrossentropy on the labels cast to `int`

Trained the model 3 epochs on the train set then 2 epochs on the validation set, after that 2 epochs on the test set with pseudo-labels.

### Datasets
- 1st model [dataset creation](https://github.com/dimitreOliveira/Jigsaw-Multilingual-Toxic-Comment-Classification/blob/master/Datasets/jigsaw-data-split-roberta-192-ratio-2-upper.ipynb) 1:2 toxic to non-toxic samples, 400830 total samples.
- 2nd model [dataset creation](https://github.com/dimitreOliveira/Jigsaw-Multilingual-Toxic-Comment-Classification/blob/master/Datasets/jigsaw-data-split-roberta-192-ratio-1-clean-polish.ipynb) 1:1 toxic to non-toxic samples, 267220  total samples.

Both models used a similar dataset, upper case text, just a sample of the negative data, data cleaning was just removal of `numbers`, `#hash-tags`, `@mentios`, `links`, and `multiple white spaces`. Tokenizer was `AutoTokenizer.from_pretrained(`jplu/tf-xlm-roberta-large', lowercase=False)` like many more.

### Inference
For inference, I predicted the `head`, `tail`, and a mix of both `(50% head &amp; 50%tail)` sentence tokens.
