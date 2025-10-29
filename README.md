# NLP Group Project

This is our repository for the **_Group 6_** Course Project for the Course: Natural Language Processing.

Dataset picked: [TL-DR](https://huggingface.co/datasets/trl-lib/tldr)

## AIM
We have decided to analyze the TL-DR Dataset and build a model that would understand the TL-DR Dataset and would take in Youtube Transcripts and preferably any other of its relevant data to provide us some good insights. In this Repo., we are currently analyzing the datasets and see where we can get ourselves into further.

## Setup

1. you can create conda env: `conda env create -f environment.yml` or use your own.
2. Please make sure to install torch separately with this command: ` uv pip install torch torchvision torchaudio accelerate --index-url https://download.pytorch.org/whl/cu121` or any other that would allow us to use CUDA.

### Update Packages
if you want to share the new packages 
you can try:  
**powershell**: `conda env export --no-builds | findstr -v "prefix:" > environment.yml`  
**bash**: `conda env export --no-builds | grep -v "prefix:" > environment.yml`

and rest of us could this `conda env update -f environment.yml --prune`

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
