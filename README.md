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

4) Download the trained model or models that you want to use - each model is provided as a `tar.gz` file which should be unpacked in this folder, e.g. `curl -L <download_url> | tar xvzf -`

 - [Ensemble model](https://github.com/GateNLP/StanceClassifier/releases/download/v0.2/model-ensemble.tar.gz): an English-language feature-based ensemble model built with Logistic Regression, Random Forest and Multi-Layer Perceptron classifiers
 - [English BERT-based model](https://github.com/GateNLP/StanceClassifier/releases/download/v0.2/model-bert-english.tar.gz): Monolingual English BERT-based model, i.e. fine-tuning of BERT for the rumour stance classification task, using threshold moving for imbalanced data treatment
 - [Multilingual BERT-based model](https://github.com/GateNLP/StanceClassifier/releases/download/v0.2/model-bert-multi.tar.gz): Multilingual version of the BERT-based model, tuned on the same English data as the above but based on the [multilingual BERT](https://github.com/google-research/bert/blob/master/multilingual.md) underlying model

## Usage

### Basic usage
```
python -m StanceClassifier -l <LANGUAGE> -s <ORIGINAL_JSON> -o <REPLY_JSON> -c <MODEL>
```
Supported model names for the `-c` option: `ens` (ensemble), `bert-english` (English BERT model), `bert-multilingual` (multilingual BERT model)

The output is a class:
 - 0.0 = support
 - 1.0 = deny
 - 2.0 = query
 - 3.0 = comment

and a vector with the probabilities returned for each class.

The folder `examples` contains examples of original tweets and replies:
 - original_old and reply_old are examples of the old JSON files (140 characters)
 - original_new and reply_new are examples of the new JSON files (280 characters)

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

In addition, the `docker` directory contains configuration to build a Docker image running a particular model of the classifier as an HTTP endpoint compliant with the [API specification](https://european-language-grid.readthedocs.io/en/release1.1.2/all/A2_API/LTInternalAPI.html) of the [European Language Grid](https://www.european-language-grid.eu).

### Training new models
To train new models, you can edit `train_model.py` (more support will be given in the future). To run:
```
python train_model.py
```
