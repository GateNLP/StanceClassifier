# StanceClassifier
Stance Classifier for the WeVerify project, to determine the stance (support, deny, question, comment) of a "reply" tweet or other social media post towards the original "target" post to which it is replying.

## Available models

There are two versions of the stance classifier available:

- "target aware" - a mode that uses both the reply post _and_ the target post together to determine the stance.  This uses two models fine-tuned from [vinai/bertweet-base](https://huggingface.co/vinai/bertweet-base), one for just the reply and one for the target and reply together
- "target oblivious" - a mode that uses just the reply post, without reference to the target.  This model is fine tuned from [cardiffnlp/twitter-xlm-roberta-base](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base), which is a multilingual model, but the fine tuning data is still the RumourEval 2019 data set which is English only.

## Usage

### Basic usage
```
python -m StanceClassifier reply.json [original.json]
```

The reply and (if provided) original arguments should be JSON files containing tweets in JSON format with at least a `"text"` or `"full_text"` property containing the text.  If only a reply is provided then the target-oblivious model will be used, if an original tweet is provided as well then the ensemble model will be used that combines a target-oblivious and a target-aware model and picks the best classification.

The output is a class:
 - 0.0 = support
 - 1.0 = deny
 - 2.0 = query
 - 3.0 = comment

and a vector with the probabilities returned for each class.

The folder `examples` contains examples of original tweets and replies:
 - original_old and reply_old are examples of the old JSON files (140 characters)
 - original_new and reply_new are examples of the new JSON files (280 characters)

### Programmatic usage

The project provides two main classes, `StanceClassifer` for target-oblivious stance detection and `StanceClassifierEnsemble` for the target-aware ensemble model.  Both classes can be imported `from StanceClassifier.stance_classifier`, and will download their models from HuggingFace on first use.

### Server usage

The `docker` directory contains configuration to build a Docker image running a particular model of the classifier as an HTTP endpoint compliant with the [API specification](https://european-language-grid.readthedocs.io/en/release1.1.2/all/A2_API/LTInternalAPI.html) of the [European Language Grid](https://www.european-language-grid.eu).

### Training new models
To train new models, you can edit `train_model.py` (more support will be given in the future). To run:
```
python train_model.py
```
