# MistralRO-finetune
Fine-tuning of a Mistral 7B model for QA in Romanian and additionally for text summarization in Romanian

The Jupyter notebook is based on the Unsloth notebook https://github.com/unslothai/unsloth

Unsloth was used to accelerate runtime. The base model is a 4 bit quantized version of Mistral. It was finedtuned using LoRA (https://arxiv.org/abs/2106.09685)

## Access the model and training set

* The model is open access, you can load it from: https://huggingface.co/OctavianB/MistralRo
* For the version finetuned on Romanian text summarization: https://huggingface.co/OctavianB/MistralRoSummary
* The training set used to fine-tune for Romanian can also be accessed here: https://huggingface.co/datasets/fcolt99/combined_dataset

## Training

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



## Testing

The test samples were extracted from the romanian text summarization dataset, as a very small subset due to the slow inference time of the models. Model performance was evaluated using BERTScore, which is defined as the cosine similarity between the BERT embeddings of the input and the output text.


  

