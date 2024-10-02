# GIF-QA

## Introduction

GIF Question Answering is a multimodal task in machine learning that aims to answer a given question by making use of a provided video as the context. This repository presents a detailed attempt at solving this task
by making use of the TGIF dataset by imbibing spatio-temporal knowledge inside Large Language Models. Various facets of the problem such as dataset generation, GIF frame processing and finally the model architecture are 
documented here. For the entire report, kindly refer to the report.pdf file included above.

## Usage

Download the model weights from the drive link mentioned in ckpt.md in the checkpoints folder. Then, for evaluation kindly use the notebook provided in eval.

## Synthetic Data Generation

For the task at hand, it was important to first develop a question-answer pair dataset using synthetic methods, as manual annotation would be too time consuming. After a thorough
search, we came up with an efficient pipeline to generate a decent set of synthetic question-answer pairs from the given dataset, more specifically, the descriptions.
We employ the T5-base model fine-tuned on SQuAD from HuggingFace for Question Generation. Our exact approach follows the steps listed below :

1. Consider each url, and the corresponding description.
2. Extract potential answers from the description, making use of the spacy library for
parts-of-speech tagging. The broad categories include:
  a. living entities
  b. nouns
  c. verb
  d. prepositional phrases (eg. in a box)
  e. colors
3. Supply these potential answers to the pipeline, and retrieve question-answer-url triplets.
   
Using this approach, we were able to generate 72k question-answer pairs (owing to time
constraints, we were unable to generate QA pairs from all the GIFs)

## GIF Frame Processing Pipeline

![clipcut drawio](https://github.com/user-attachments/assets/cfc6dcb9-0bb4-4d78-974a-cada3e36980a)

Cutting down on the number of frames cleverly can allow us to increase efficiency and also possibly introduce other multimodal LLMs into the picture, at least for valuable inference on these frames, while not
losing out on valuable information or including irrelevant frames. In this light, we propose CLIP-Cut, an efficient pipeline for keyframe extraction from GIFs. CLIP-Cut is essentially based on two core pillars, 
mainly, CLIP and FAISS.

We first sample all the frames from the GIF url, and then, make use of the FAISS library to
generate an faiss-index by grouping every 3 frames together. Then, the CLIP transformer is
utilized to project the GIF frames and the text input, comprising the question and the description,
to project the language and image features into the same space, and retrieving the most similar
frames using an index search.

## Model Architecture

![hopenew](https://github.com/user-attachments/assets/6dfcbdc8-0fd3-4b60-bc3f-d7eebb5794d4)

A recent ECCV submission made by researchers at Tencent, *ST-LLM: Large Language Models Are Effective Temporal Learners* deliberated over the spatio-temporal reasoning abilities of
LLMs without explicitly defining Temporal Adapters or carrying out extensive pre-training tasks. The results of the paper were quite commendable, with the model beating several existingvideo-large language models on several VideoQA benchmarks such as MSVD-QA, MVBench,
MSVRTT-QA

Our final architecture consists of the CLIP-Cut retriever that returns the top-k frames relevant tothe question. These frames are then passed to a CLIP vision encoder, followed by the
pre-trained Q-Former of BLIP2. We then use a linear layer to project the image features to thedimensions of the LLM input. The question is also tokenized, and we simply prepend the frame
features before the question, and let our model predict the response.


### New Loss function for improved training

We implemented the Masked Video Modelling loss that was described in the paper. The masked language modeling loss aims to improve the spatio-temporal understanding of LLMs while simultaneously using its semantic powers. During training, we randomly mask a certain sample of the frame tokens that have been generated. Using these masked tokens, we generate an output after a forward pass through the language head, and label it fllm(I) Then, we conduct an extra forward pass through the language model, and label this outcome as fllm(I’).

![eqn](https://github.com/user-attachments/assets/6da7178d-9486-4961-886b-1b0908ac182b)

We then consider the last hidden state of the LLM, and the loss is then computed as the RMSE loss obtained from the unmasked positions of the image tokens, i.e,





