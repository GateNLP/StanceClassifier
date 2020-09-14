# StanceClassifier
Stance Classifier for the WeVerify project

This is a re-implementation of Aker et al. (2017) ["Simple Open Stance Classification for Rumour Analysis"](https://arxiv.org/pdf/1708.05286.pdf). We replaced the Bag-of-words and BROWN features with [GloVe embeddings](https://nlp.stanford.edu/projects/glove/) (twitter embeddings with 200 dimensions). 

## Configuration
1) Requirements:

        python3.7
        nltk
        numpy
        scipy
        sklearn
    
        Get Vader lexicon: 
        `python -m nltk.downloader vader_lexicon`

2) Clone this repository

3) Download the [resources](https://github.com/GateNLP/StanceClassifier/releases/download/v0.1/resources.tar.gz) required for feature extraction and extract it inside the main folder (`StanceClassifer`)

4) Download the [trained models](https://github.com/GateNLP/StanceClassifier/releases/download/v0.1/models.tar.gz) archive and extract it inside the main folder (`StanceClassifer`)

## Usage

### Basic usage
```
python -m StanceClassifier -l <LANGUAGE> -s <ORIGINAL_JSON> -o <REPLY_JSON> -c <MODEL>
```
    Supported languages: en
    Supported models: 
        ens (Feature-based ensemble model built with Logistic Regression, Random Forest and Multi-Layer Perceptron classifiers)
        bert-tm (BERT-based model, i.e. fine-tuning of BERT for the rumour stance classification task, using threshold moving for imbalanced data treatment)

    The output is a class:
        0.0 = support
        1.0 = deny
        2.0 = query
        3.0 = comment

    and a vector with the probabilities returned for each class.

    The folder `examples` contains examples of original tweets and replies:
        original_old and reply_old are examples of the old JSON files (140 characters)
        original_new and reply_new are examples of the new JSON files (280 characters)

### `StanceClassifer` class (StanceClassifier.stance_classifier.StanceClassifier)
This is the main class in this project. If you want to add this project as part of your own project, you should import this class. 

### Server usage
We have implemented TCP and HTTP servers. Server parameters are defined in the `configurations.txt` file.

To run the TCP server:
```
python Run_TCP_StanceClassifier_Server.py
```

Testing the TCP server:
```
python Test_Run_TCP_StanceClassifier_Server.py
```

The HTTP server uses a TCP server already running:
```
python Run_HTTP_StanceClassifier_Server.py
```

To test the HTTP server:
```
python Test_Run_HTTP_StanceClassifier_Server.py
```

### Training new models
To train new models, you can edit `train_model.py` (more support will be given in the future). To run:
```
python train_model.py
```
