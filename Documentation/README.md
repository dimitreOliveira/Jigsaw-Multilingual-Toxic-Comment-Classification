# Jigsaw Multilingual Toxic Comment Classification - Planning
 
## Working cycle:
1. Read competition information and relevant content to feel comfortable with the problem. Create a hypothesis based on the problem.
2. Initial data exploration to feel comfortable with the problem and the data.
3. Build the first implementation (baseline).
4. Loop through [Analyze -> Approach(model) -> Implement -> Evaluate].

## 1. Literature review (read some kernels and relevant content related to the competition).
- ### Relevant content:
  - [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
  - [The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)](https://jalammar.github.io/illustrated-bert/)
  - [HuggingFace Transformers](https://huggingface.co/transformers/)
  - [Hugging Face: State-of-the-Art Natural Language Processing in ten lines of TensorFlow 2.0](https://medium.com/tensorflow/using-tensorflow-2-for-state-of-the-art-natural-language-processing-102445cda54a)
  - [Simple BERT using TensorFlow 2.0](https://towardsdatascience.com/simple-bert-using-tensorflow-2-0-132cb19e9b22)
  - [TF Hub - bert_en_uncased_L-12_H-768_A-12](https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1)
  - [Multilingual Universal Sentence Encoder for Semantic Retrieval](https://ai.googleblog.com/2019/07/multilingual-universal-sentence-encoder.html)
  - [BERT Word Embeddings Tutorial](https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/)
  - [A Visual Guide to Using BERT for the First Time](https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)
  - [Multi-Sample Dropout in Keras](https://towardsdatascience.com/multi-sample-dropout-in-keras-ea8b8a9bfd83)
  - [Jigsaw Toxicity Classification 1st place solution presentation](https://docs.google.com/presentation/d/1Km1C8jKpofoRRnWRMka-SGrfRNd8jlp6TuasqVKY91E/edit#slide=id.p)
  - [XLM — Enhancing BERT for Cross-lingual Language Model](https://towardsdatascience.com/xlm-enhancing-bert-for-cross-lingual-language-model-5aeed9e6f14b)

- ### GitHub:
  - [Huggingface transformers](https://github.com/huggingface/transformers)
  - [Hggingface tokenizers](https://github.com/huggingface/tokenizers/tree/master/bindings/python)

- ### Papers:
  - [How to Fine-Tune BERT for Text Classification?](https://arxiv.org/abs/1905.05583)
  - [Multi-Sample Dropout for Accelerated Training
and Better Generalization](https://arxiv.org/pdf/1905.09788.pdf)

- ### Videos:
  - [BERT for Kaggle Competitions | Yuanhae Wu | Kaggle Days](https://www.youtube.com/watch?v=jS79Y8I0DF4&t=9s)
  - [Deep Learning Formulas for NLP Applications | Chenglong Chen | Kaggle](https://www.youtube.com/watch?v=SmsAI0kLJFc&t=0s)
  - [Solving NLP Problems with BERT | Yuanhao Wu | Kaggle](https://www.youtube.com/watch?v=rQQAIJIf60s)
  - [BERT Research by Chris McCormick](https://www.youtube.com/watch?v=FKlPCK1uFrc&list=PLam9sigHPGwOBuH4_4fr-XvDbe5uneaf6)

- ### Kernels:
  - [Jigsaw TPU: DistilBERT with Huggingface and Keras](https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras)
  - [Jigsaw TPU: XLM-Roberta](https://www.kaggle.com/xhlulu/jigsaw-tpu-xlm-roberta/notebook)
  - [Understanding cross lingual models](https://www.kaggle.com/mobassir/understanding-cross-lingual-models/notebook)

- ### Discussions:
  - [Everything you always wanted to know about BERT (but were afraid to ask)](https://www.kaggle.com/c/google-quest-challenge/discussion/128420)
  - [BERT & Friends Reference](https://www.kaggle.com/c/tensorflow2-question-answering/discussion/126702)
  - [1ST PLACE SOLUTION - Jigsaw competition](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/103280#latest-619135)
  - [Let's Complicate Things [Bert]](https://www.kaggle.com/c/google-quest-challenge/discussion/123770)
  - [Past (Kaggle) Jigsaw Competitions & Top Solutions](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/138163)
  - [Insights on achieving 0.9383 without translating data](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/140254)
  - [Good Papers on Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/144097)
  - [Cheatsheet - Text Preprocessing](https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/146888)
 
- ### Insights:
  - #### Positive Insights
    - Test set seems to have a 1:5 ratio between positive:negative samples
  - #### Negative Insights
    - Competition labels are heavily unbalanced
    - Train data has only english text
    - Validation data has only text on 3 languages while test has 6 languages
