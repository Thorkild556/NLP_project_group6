# NLP Group Project - Youtube Information Extraction 

This is our repository for the **_Group 6_** Course Project for the Course: Natural Language Processing.

Dataset picked: [TL-DR](https://huggingface.co/datasets/trl-lib/tldr)

# P2 updated for P3:

## Where to find analysis
You can find the initial analysis we did for the TL;DR dataset inside the lab/ExploreTLDR.ipynb. The other notebooks are for testing the data pipeline for our project and has nothing to do with the analysis of project milestone p2

## Abstract
With this project we aim to contribute with a new method for information extraction using youtube video transcripts. Our goal is to create a framework that takes in a prompt, fetches information from youtube videos, and returns compressed information that aims to answer the prompt and makes references to the videos. For the creation of this framework we first plan on using the TL;DR dataset + a custom dataset with youtube transcript and wanted outputs for model finetuning. The TL;DR will be formatted to be more alligned with the custom dataset. The goal is to then create a model that can generate a summary of multiple youtube videos. The intention of this is that this can take more perspectives and nuances into account than the more concise information that might be contained eg. in a single page of information or a single video. In the project we test 3 different models to test which methods works best for the task to be solved. One baseline model with instructions, another finetuned on a custom gold standard data-set and the last one tuned on a reformatted TL;DR data-set, as well as the custom data-set. The TL;DR dataset is used for the purpose that it has more data and we might be able to exploit this by using the concept of transfer learning.

## Contribution
The contribution we are aiming to achieve, is to provide a new method for information extraction that uses multiple transcripts of youtube videos and then summarizing them according to the prompt. This method contributes a new idea that tries to improve on information extraction and question answering by using text extracted from more naturally spoken language, making it different from usual document retrieval and could possibly improve on making the information extraction more human. This could be the case as using multiple youtube videos, eg. in comparison to a single, might contain more explanations, perspectivations and multiple different nuances rather than what a single video or page of information might express. This could potentially compress information into a more human answer, putting the information into a broader perspective. It might also reveal if there seems to be any contradictions between multiple transcripts making the output more trustworthy.

## Additional dataset
Apart from using the TL;DR, we will also be constructing an additional dataset that specifically matches multiple youtube transcripts, merging the information into a single summary. We will construct this dataset by having a larger language model (GPT 5.1) generate summaries that takes the nuances of each of 2-4 videos into account and puts out a wanted summary. We will have to pre-prompt it with the precise instructions that will make it output the type of summary we want. In this way after using the TL;DR we will be using a kind of distilation of our summary model making it specifically good at this exact summary task. This would be good as we want the model to have some different properties like referencing which video says what and say when contradictions happen.

## Methods 
### Model training and pipeline creation 
To achieve the goal framework, we will be using a pretrained model (Llama 3.2, 3B, 8-bit), as it has some language understanding, as it is already a pretrained seq2seq text generation model. Using this model we will first be fine-tuning it on the, formatted post and TL;DR pairs from the reddit TL;DR dataset. This will finetune the model for multiple text summarization. Now our model will have multiple text summary capabilities, which we expect will make transfer learning into our specific summarization task easier.

Using the custom additional dataset, we want to further finetune our summary model. This will make the model more specific for our task. This way we have created the main model that our framwork is build upon.

For the implementaion of our full framework, the interface will be simple. The user will ask a question, the question is then prompted to youtube as a youtube query. The api will then fetch the transcripts of the top 4 videos. The transcripts will then be tokenized and fed into the model which should then generate the summary, which is then returned back to the user together with the links to the videoes. The links are for if the user wants more information.

This is the very base of our idea which, can potentially be improoved upon. If time allows, we would like to further improve the framework by making another model that takes in a natural language question and then produces a youtube search query, that finds videos that answers this question. This would be different from the user having to formulate the youtube search query them self, as youtube search queries are not neccesarily always posed as a question using natural language. Youtube already does this to some degree, but it would be a small quality of life improvement making it easier to interact with the framework. 

### Model evaluation
Model evaluation will be done on a seperate test sample of the dataset that we created as it is the only dataset that has the seq2seq properties that we want the model to have. The evaluation metrics used will be BERT score between the actual generation of the model and the wanted text generation as defined by either us or the larger model as described in the "aditional dataset section". We will be using BERT score instead of classical measures since we do not neccesarily care about whether the exact text is the same but whether the semantic meaning of the generation is close to the message we want the model to generate. 

## A summary of the updates from P2

Instead of using the raw TL;DR we decided to make a version that is closer alligned with the gold standard we are trying to achieve. Furthermore, to make sure that this process does actually make sense we also compare it to a baseline model and a model, only fine-tuned on the custom "gold standard" data-set. We used the a Tukey's statistical test to compare the differences of their respective bert-scores

## Proposed timeline & internal milestones

14th november: The api and data import should be ready and useable

21th november: Model should be fully finetuned on the TL:DR data

28th november: Creation of the custom data set should be finished.

5th december: Last class and the overall code is done and the model is finetuned

12th december: All code is done and only paper writing is left

19th december: Hand in project

## Questions for instructor:
- Where can we access the cloud computing resources and how does it work?

## Setup of the project

1. Create Python Venv and Activate it
2. Install poetry: `pip install poetry`
3. before you start installation of the packages you might need to install: [Rust](https://rustup.rs/) you can skip it if already installed (it is required by bert-score package)
4. Then simply do `poetry install`


## Only for the data pipeline testing (not part of P2)
### Update Packages
you can add from this [add command](https://python-poetry.org/docs/cli/#add).

## How to Use it ?

if you want to use the lab notebooks please follow the files inside the `lab/`
you can maybe download them and run it in colab. we would be having many more folders as we go for final application.

# P3:
