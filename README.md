# MistralRO-finetune
The following is a fine-tuning of a Mistral 7B model for QA in Romanian and additionally for text summarization in Romanian

The Jupyter notebook is based on the Unsloth notebook https://github.com/unslothai/unsloth

Unsloth was used to accelerate runtime. The base model is a 4 bit quantized version of Mistral. It was finetuned using LoRA (https://arxiv.org/abs/2106.09685)

## Access the model and training set

* The romanian model is open access, you can load it from: https://huggingface.co/OctavianB/MistralRo
* For the version finetuned on Romanian text summarization: https://huggingface.co/OctavianB/MistralRoSummary
* The training set used to fine-tune for Romanian can also be accessed here: https://huggingface.co/datasets/fcolt99/combined_dataset

## Training sets

To give the model a better understanding of romanian before using it for text summarization, a translated QA dataset was used for fine-tuning. A second round of fine-tuning on a romanian text summarization dataset was realized to improve its performance on this specific task.
The  [OpenAssistant Conversations Dataset (oasst1)](https://huggingface.co/datasets/readerbench/ro-text-summarization) dataset containing 84400 text samples was used.
The set contains text in multiple languages and was translated by [opus translation model](https://huggingface.co/Helsinki-NLP/opus-mt-en-ro).
The most important columns to our finetuning task are text, role, and rank:
*	The text rank has the content that the model will be trained on
*	Role is used when training mistral, since it expects a dataset of alternating Q and A text prompts from „user” and „assistant”
*	Rank is used for creating the Q and A pairs. We can determine which questions belong to which answers by using this column, since every answer after a question is given ranks starting from 0

All the samples were used. The reasoning behind the use of this dataset was to give the model a better understanding of the Romanian language before using it for the text summarization task.

The [readerbench/ro-text-summarization](https://huggingface.co/datasets/readerbench/ro-text-summarization) dataset contains 65.3k samples on the training set and 7.25k samples on the test set. It contains summaries of text obtained from various news outlets. Only the ‘Content’ and ‘Summary’ fields were used:
*	Content contains the raw text
*	Summary is the target output for said content

## Training pipeline

The overall outline of the training pipeline was the following:
*	The OAST1 dataset is translated into Romanian using the OPUS1 model
*	The Mistral7b instruct model is finetuned using LoRA and PEFT on the translated dataset to increase the model's ability to speak Romanian (takes ~10 minutes on the T4 GPU on free tier Google Colab using Unsloth
*	The finetuned model is then trained on the ro-text-summarization dataset to obtain the second version of the finetune.

<!--
Below are the training loss graphs for the two finetuned models (romanian finetuning on left, romanian text summarization finetuning on right):

<div style="display: flex;" align="center">
  <img src="./figs/Loss Romanian finetune.jpg" style="width: 40%;">
  <img src="./figs/Loss Romanian finetune.jpg" style="width: 40%;">
</div>
-->

## Evaluation


The test samples were extracted from the romanian text summarization dataset, as a very small subset due to the slow inference time of the models. Model performance was evaluated using BERTScore, which is defined as the cosine similarity between the BERT embeddings of the input and the output text.

We used the first n = 10 samples of the ro-text-summarization dataset to test each of the three models in conjunction with the BERT score to evaluate the text generations (relative to the ground-truth). The test sample size was small because of the limited resources at hand. 
Below is a table containing the average of the three metrics given by the BERT score (precision, recall, f1) for each model over the ten samples.

<div align="center">
  
| Metric | Mistral 7B | Mistral 7B Ro | Mistral 7B Ro Summary |
|----------|:----------:|:----------:|:----------:|
| Precision | 0.640 | 0.665 | 0.652 |
| Recall | 0.670 | 0.690 | 0.679 |
| F1 | 0.678 | 0.678 | 0.676 |

</div>
  

