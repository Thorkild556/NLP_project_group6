# NLP Group Project - Youtube Information Extraction

This is our repository for the **_Group 6_** Course Project for the Course: Natural Language Processing.

Dataset picked: [TL-DR](https://huggingface.co/datasets/trl-lib/tldr)

## Abstract
With this project we aim to contribute with a new method for information extraction using youtube video transcripts. Our goal is to create a framework that takes in a prompt, fetches information from youtube videos, and returns compressed information that aims to answer the prompt and makes references to the videos. For the creation of this framework we first plan on using the TL;DR dataset + a custom dataset with youtube transcript and wanted outputs for model finetuning. The goal is to then have a summary of all the videos that is more in line with human language and information dissemination. This is as more youtube videos could take more perspectives and nuances into account than the more precise and concise information that might be contained in something like a wikipedia page.

## Contribution
The contribution we are aiming to achieve, is to provide a new method for information extraction that uses multiple transcripts of youtube videos and then summarizing them according to the prompt. This method contributes a new idea that tries to improve on information extraction by using text extracted from more naturally spoken language, making it a different from usual document retrieval and possibly improve on making the information extraction more human. This could be the case because using a couple youtube videos, eg. in comparison to wikipedia, might contain more explanations and perspectivations multiple different nuances rather than the concise version of information that a wikipedia page might express. This could potentially compress information into a more human answer, putting the information into a broader perspective.

## Additional dataset
Apart from using the TL;DR, we will also be constructing an additional dataset that specifically matches multiple youtube transcripts, merging the information into a single summary. We will construct this dataset by having a larger language model generate summaries that takes the nuances of each video into account and puts out a wanted summary. We will have to pre-prompt it with the precise instructions that will make it output the type of summary we want. In this way after using the TL;DR we will be using a kind of distilation of our summary model making it specifically good at this exact summary task. Potentially we could also ourselves write a few summary examples to fine tune the model on, apart from the larger language model generations. This would be good as we want the model to have some different properties like referencing which video says what and potentially where they differ.

## Methods 
### Model training and pipeline creation 
To achieve the goal framework, we will be using a pretrained model like T5, as it has some language understanding, as it is already pretrained seq2seq text generation. Using the T5 model we will then be finetuning it for seq2seq summarization, using the prompt and TL;DR pairs from the TL;DR dataset. This will further finetune the T5 model for summarization. Now our model will have text summary capabilities, which we expect will make transfer learning into our specific summarization task easier, as we now have a summary model. 

Now once we have this summary model, we can start finetuning it for our specific usecase. For this we will need a self made dataset as described in the above section.
The youtube transcripts will be extracted using a youtube transcript API. We will then personally make some different youtube search queries and extract the top 4 videos given those queries. Then we will note down these prompt and video-transcipts pairs into a dataset. Once we have made a few hundreds of such pairs, we will now create the answers that we want our model to generate. This will be done as described in the "Aditional dataset" section above, and add it to our dataset.

Using this new dataset, we now want to further finetune our summary model on our dataset. This will make the model more specific for our task. This way we have created the main model that our framwork is build upon.

For the implementaion of our full framework, the interface will be simple. The user will ask a question, the question is then prompted to youtube as a youtube query. The api will then fetch the transcripts of the top 4 videos. The transcripts will then be tokenized and fed into the model which should then generate the summary, which is then returned back to the user together with the links to the videoes. The links are for if the user wants more information.

This is the very base of our idea which, can potentially be improoved upon. If time allows, we would like to further improve the framework by making another model that takes in a natural language question and then produces a youtube search query, that finds videos that answers this question. This would be different from the user having to formulate the youtube search query them self, as youtube search queries are not neccesarily always posed as a question using natural language. Youtube already does this to some degree, but it would be a small quality of life improvement making it easier to interact with the framework. 

### Model evaluation
Model evaluation will be done on a seperate test sample of the dataset that we created as it is the only dataset that has the seq2seq properties that we want the model to have. The evaluation metrics used will be BERT score between the actual generation of the model and the wanted text generation as defined by either us or the larger model as described in the "aditional dataset section". We will be using BERT score instead of classical measures since we do not neccesarily care about whether the exact text is the same but whether the semantic meaning of the generation is close to the message we want the model to generate. 

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
3. Then simply do `poetry install`

### Update Packages
you can add from this [add command](https://python-poetry.org/docs/cli/#add).

## How to Use it ?

if you want to use the lab notebooks please follow the files inside the `lab/`
you can maybe download them and run it in colab. we would be having many more folders as we go for final application.

## Lab Files

Following are the files and what they are for:

### ExploreTLDR.ipynb

In this file we would be analyzing the TL-DR Dataset and see for any patterns, we have plotted and transformed datasets. you can give it a read for more details.

### ForGettingAVideoDetails.ipynb

In this file we would be seeing how we could fetch the data for a youtube video and what can we fetch a video like video details and its transcripts.

### GuidelinesForProject-Google.ipynb

In order to start with [ForGettingAVideoDetails.ipynb](#forgettingavideodetailsipynb) you would need to create a project in the Google Cloud for accessing Youtube Data API. so follow these steps documented here.
