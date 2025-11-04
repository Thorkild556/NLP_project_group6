# NLP Group Project

This is our repository for the **_Group 6_** Course Project for the Course: Natural Language Processing.

Dataset picked: [TL-DR](https://huggingface.co/datasets/trl-lib/tldr)

## AIM
We have decided to analyze the TL-DR Dataset and build a model that would understand the TL-DR Dataset and would take in Youtube Transcripts and preferably any other of its relevant data to provide us some good insights. In this Repo., we are currently analyzing the datasets and see where we can get ourselves into further.

## Setup

1. Create Python Venv and Activate it
2. Install poetry: `pip install poetry`
3. before you start installation of the packages you might need to install: [Rust](https://rustup.rs/) you can skip it if already installed (it is required by bert-score package)
4. Then simply do `poetry install`

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
