# NLP Group Project - YouTube Information Extraction 

This is our repository for the **_Group 6_** Course Project for the Course: Natural Language Processing.

Dataset picked: [TL-DR](https://huggingface.co/datasets/trl-lib/tldr)

# P2 updated for P3:

## Abstract
With this project, we aim to contribute a new method for information extraction using YouTube video transcripts. Our goal is to create a framework that takes in a prompt, fetches information from YouTube videos, and returns compressed information that aims to answer the prompt and makes references to the videos. For the creation of this framework, we first plan on using the TL;DR dataset + a custom dataset with YouTube transcripts and wanted outputs for model finetuning. The TL;DR will be formatted to be more alligned with the custom dataset (see [`export_dataset.ipynb`](https://github.com/au-nlp/project-milestone-p2-group-6/blob/main/lab/export_dataset.ipynb)). The goal is to then create a model that can generate a summary of multiple YouTube videos (see [`fine_tune_models.ipynb`](https://github.com/au-nlp/project-milestone-p2-group-6/blob/main/lab/fine_tune_models.ipynb)). The intention of this is that this can take more perspectives and nuances into account than the more concise information that might be contained e.g. in a single page of information or a single video. In the project, we test 3 different models to test which method works best for the task to be solved (see [`model_evaluation.ipynb`](https://github.com/au-nlp/project-milestone-p2-group-6/blob/main/lab/model_evaluation.ipynb)). One baseline model with instructions, another finetuned on a custom gold standard data-set, and the last one tuned on a reformatted TL;DR data-set, as well as the custom data-set. The TL;DR dataset is used for the purpose that it has more data, and we might be able to exploit this by using the concept of transfer learning.

## Contribution
The contribution we are aiming to achieve is to provide a new method for information extraction that uses multiple transcripts of YouTube videos and then summarizes them according to the prompt. This method contributes a new idea that tries to improve on information extraction and question answering by using text extracted from more naturally spoken language, making it different from usual document retrieval and could possibly improve on making the information extraction more human. This could be the case as using multiple YouTube videos, e.g. in comparison to a single, might contain more explanations, perspectives, and multiple different nuances rather than what a single video or page of information might express. This could potentially compress information into a more human answer, putting the information into a broader perspective. It might also reveal if there seems to be any contradictions between multiple transcripts making the output more trustworthy.

## Additional dataset
Apart from using the TL;DR, we will also be constructing an additional dataset that specifically matches multiple YouTube transcripts, merging the information into a single summary (see [`fetch_custom_dataset.py`][custom-fetch]). We will construct this dataset by having a larger language model (GPT 5.1) generate summaries that takes the nuances of each of 2-4 videos into account and puts out a wanted summary (see [`summary_for_custom_dataset.py`][custom-gpt]). We will have to pre-prompt it with the precise instructions (see [`instructions.txt`][custom-instr]) that will make it output the type of summary we want. In this way after using the TL;DR we will be using a kind of distilation of our summary model making it specifically good at this exact summary task. This would be good as we want the model to have some different properties like referencing which video says what and say when contradictions happen.

[custom-fetch]: https://github.com/au-nlp/project-milestone-p2-group-6/blob/main/src/fetch_custom_dataset.py
[custom-gpt]: https://github.com/au-nlp/project-milestone-p2-group-6/blob/main/src/summary_for_custom_dataset.py
[`instructions.txt`]: https://github.com/au-nlp/project-milestone-p2-group-6/blob/main/data/instructions.txt
## Methods 
### [Model training](https://github.com/au-nlp/project-milestone-p2-group-6/blob/main/lab/fine_tune_models.ipynb)
To achieve the goal framework, we will be using a pretrained model (Llama 3.2, 3B, 8-bit), as it has some language understanding, as it is already a pretrained seq2seq text generation model. Using this model we will first be fine-tuning it on the formatted post and TL;DR pairs from the reddit TL;DR dataset. This will finetune the model for multiple text summarization. Now our model will have multiple text summary capabilities, which we expect will make transfer learning into our specific summarization task easier.

Using the custom additional dataset, we want to further finetune our summary model. This will make the model more specific for our task. This way we have created the main model that our framework is build upon.

### [Model evaluation](https://github.com/au-nlp/project-milestone-p2-group-6/blob/main/lab/model_evaluation.ipynb)
Model evaluation will be done on a separate test sample of the dataset that we created as it is the only dataset that has the seq2seq properties that we want the model to have. The evaluation metrics used will be BERT score between the actual generation of the model and the wanted text generation as defined by either us or the larger model as described in the "additional dataset section". We will be using BERT score instead of classical measures since we do not necessarily care about whether the exact text is the same, but whether the semantic meaning of the generation is close to the message we want the model to generate. 

## A summary of the updates/changes since P2

Instead of using the raw TL;DR we decided to make a version that is closer alligned with the gold standard we are trying to achieve as explained in the report. Furthermore, to make sure that this "transfer learning" process does actually make sense we also changed the model evaluation to further compare it to a baseline model and a model, only fine-tuned on the custom "gold standard" data-set. We used the Tukey's statistical test to compare the differences of their respective bert-scores.

# Contributions:

#### Thorkild Kappel:
Thorkild contributed with how the system should be designed in order to achieve the goal we set out to do. He wrote the code for the preprocessing and formatting of the TL;DR data as well as the final analysis and evaluation of the models. He co-wrote the code for model finetuning together with Rahul. Thorkild also wrote everything in the report, as well as the appendix equations, except for the introduction and motivation. However, everyone read the paper through.

#### Rahul
Rahul contributed with building the pipeline for extracting the custom dataset (from YouTube) and also generating the summaries from azure foundry's gpt 5.1 agent. co-wrote the code for finetuning the models with Thorkild, and also wrote code to generate the output from the fine-tuned models with the samples we have. Rahul also handled refracting of our code, re-arranging files in our project as well as handle any cloud based execution tasks along with Jolin.

#### Zhuolin "Jolin" Li
Jolin contributed with exploratory data analysis and basic visualization. She helped spot check the custom dataset formatting. She also participated in training and testing the models helped identify issues during experimentation, and assisted with workflow adjustments. In addition, she wrote the introduction and motivation section of the report, and reviewed related literature. She also supported the team with shared Colab coordination and cloud-based execution tasks.
