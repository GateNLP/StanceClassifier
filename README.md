# StanceClassifier
Stance Classifier for the WeVerify project

This is a re-implementation of Aker et al. (2017) ["Simple Open Stance Classification for Rumour Analysis"](https://arxiv.org/pdf/1708.05286.pdf). We replaced the Bag-of-words and BROWN features with [GloVe embeddings](https://nlp.stanford.edu/projects/glove/) (twitter embeddings with 200 dimensions). 

## Configuration
    1) Requirements:
    
        nltk
        numpy
        scipy
        sklearn
    
        Get Vader lexicon: `python -m nltk.downloader vader_lexicon`

    2) Clone this repository

    3) Download the [resources](http://staffwww.dcs.shef.ac.uk/people/C.Scarton/resources.tar.gz) required for feature extraction and extract it inside the main folder (`StanceClassifer`)

## Usage

### Basic usage
```
python __main__.py -l <LANGUAGE> -s <ORIGINAL_JSON> -o <REPLY_JSON> -c <MODEL>
```
    Supported languages: en
    Supported models: 
        rf (Random Forest)
        mlp (Multi-layer perceptron)
        lr (Logistic Regression)
        svm (Support Vector Machines)

    The output is a class:
        0.0 = support
        1.0 = deny
        2.0 = query
        3.0 = comment

    and a vector with the probabilities returned for each class.

    The folder `examples` contains a source and reply example.

### `StanceClassifer` class
This is the main class in this project. If you want to add this project as part of your own project, you should import this class. 

