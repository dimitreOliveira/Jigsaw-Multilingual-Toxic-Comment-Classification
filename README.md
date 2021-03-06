![](https://github.com/dimitreOliveira/Jigsaw-Multilingual-Toxic-Comment-Classification/blob/master/Assets/banner.png)

## About the repository
This competition was about developing multi-lingual models to classify if comments from forums were toxic or not, the test set had comments on 6 different languages (English, Portugese, Russian, French, Italian and Spanish), for my experiments I manly use [XLM-RoBERTa](https://arxiv.org/abs/1911.02116) a SOTA multi-lingual model form [Huggingface](https://huggingface.co/transformers/model_doc/xlmroberta.html) repository.

### Published Kaggle kernels:
- [Jigsaw - TPU optimized training loops](https://www.kaggle.com/dimitreoliveira/jigsaw-tpu-optimized-training-loops)
- [Jigsaw Classification - DistilBERT with TPU and TF](https://www.kaggle.com/dimitreoliveira/jigsaw-classification-distilbert-with-tpu-and-tf)

### What you will find
- Best solution (Bronze medal - 100th place) [link](https://github.com/dimitreOliveira/Jigsaw-Multilingual-Toxic-Comment-Classification/tree/master/Best%20solution%20(Bronze%20medal%20-%20100th%20place))
- Datasets [[link]](https://github.com/dimitreOliveira/Jigsaw-Multilingual-Toxic-Comment-Classification/tree/master/Documentation)
- Documentation [[link]](https://github.com/dimitreOliveira/Jigsaw-Multilingual-Toxic-Comment-Classification/blob/master/Documentation/Planning.md)
- Models [[link]](https://github.com/dimitreOliveira/Jigsaw-Multilingual-Toxic-Comment-Classification/tree/master/Model%20backlog)
   - Inference [link](https://github.com/dimitreOliveira/Jigsaw-Multilingual-Toxic-Comment-Classification/tree/master/Model%20backlog/Inference)
   - Train [link](https://github.com/dimitreOliveira/Jigsaw-Multilingual-Toxic-Comment-Classification/tree/master/Model%20backlog/Train)
- Scripts [link](https://github.com/dimitreOliveira/Jigsaw-Multilingual-Toxic-Comment-Classification/tree/master/Scripts)

### Jigsaw Multilingual Toxic Comment Classification
#### Use TPUs to identify toxicity comments across multiple languages

Kaggle competition: https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification

### Overview

It only takes one toxic comment to sour an online discussion. The Conversation AI team, a research initiative founded by [Jigsaw](https://jigsaw.google.com/) and Google, builds technology to protect voices in conversation. A main area of focus is machine learning models that can identify toxicity in online conversations, where toxicity is defined as anything rude, disrespectful or otherwise likely to make someone leave a discussion. If these toxic contributions can be identified, we could have a safer, more collaborative internet.

In the previous 2018 [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge), Kagglers built multi-headed models to recognize toxicity and several subtypes of toxicity. In 2019, in the [Unintended Bias in Toxicity Classification Challenge](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification), you worked to build toxicity models that operate fairly across a diverse range of conversations. This year, we're taking advantage of [Kaggle's new TPU support](https://www.google.com/url?q=https://www.kaggle.com/docs/tpu&sa=D&ust=1584991625681000&usg=AFQjCNGJ-qx4nDeimQMPPosTeE9daz1o7A) and challenging you to build multilingual models with English-only training data.

Jigsaw's API, [Perspective](http://perspectiveapi.com/), serves toxicity models and others in a growing set of languages (see our [documentation](https://github.com/conversationai/perspectiveapi/blob/master/2-api/models.md#all-attribute-types) for the full list). Over the past year, the field has seen impressive multilingual capabilities from the latest model innovations, including few- and zero-shot learning. We're excited to learn whether these results "translate" (pun intended!) to toxicity classification. Your training data will be the English data provided for our previous two competitions and your test data will be Wikipedia talk page comments in several different languages.

As our computing resources and modeling capabilities grow, so does our potential to support healthy conversations across the globe. Develop strategies to build effective multilingual models and you'll help Conversation AI and the entire industry realize that potential.

Disclaimer: The dataset for this competition contains text that may be considered profane, vulgar, or offensive.
