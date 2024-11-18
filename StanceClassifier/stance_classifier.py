import os
import json
import sys
import numpy as np
from .features.extract_features import Features
from .testing import test
from transformers import AutoModelForSequenceClassification,AutoTokenizer



class StanceClassifier:

    def __init__(self, model="GateNLP/stance-twitter-xlm-target-oblivious", feature_extractor=None):
        if not feature_extractor:
            # Create a plain Features() instance loading its tokenizer from
            # the same place as the model
            feature_extractor = Features(model)

        self.feature_extractor = feature_extractor
        self.model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=4)
        
    
    def classify(self, reply): 	
        
        encoded_reply, _ = self.feature_extractor.extract_bert_input(reply)
        #print("stanceclassifier.classify....................", encoded_reply, encoded_source_reply)
        stance_class, stance_prob = test.predict_bertweet(encoded_reply, self.model)
			        
        return stance_class, stance_prob


class StanceClassifierWithTarget(StanceClassifier):

    def __init__(self, model="GateNLP/stance-bertweet-target-aware", feature_extractor=None):
        if not feature_extractor:
            feature_extractor = Features(model, tokenizer_kwargs={"use_fast": False}, demojize=True)

        super().__init__(model, feature_extractor)

    def classify_with_target(self, reply, target):
        encoded_reply, reply_text = self.feature_extractor.extract_bert_input(reply)
        encoded_reply_and_target, _ = self.feature_extractor.extract_bert_input(target, reply_text)

        stance_class, stance_prob = test.predict_bertweet(encoded_reply_and_target, self.model)

        return stance_class, stance_prob


class StanceClassifierEnsemble:
    """
    Ensemble classifier that runs a target-oblivious and a target-aware model against
    the same pair of posts and returns whichever prediction is more confident.
    """

    def __init__(self, to_model="GateNLP/stance-bertweet-target-oblivious", ta_model="GateNLP/stance-bertweet-target-aware", feature_extractor=None):
        self.ta_classifier = StanceClassifierWithTarget(ta_model, feature_extractor)
        # Use the same feature extractor for both classifiers, whether that's the supplied one
        # or the one that was auto-created by the ta_classifier
        self.to_classifier = StanceClassifier(to_model, self.ta_classifier.feature_extractor)


    def classify_with_target(self, reply, target):
        # run both the target oblivious and the target aware model, and return whichever gives
        # the higher score
        stance_class_to, stance_prob_to = self.to_classifier.classify(reply)
        stance_class_ta, stance_prob_ta = self.ta_classifier.classify_with_target(reply, target)
        if stance_prob_to[stance_class_to] > stance_prob_ta[stance_class_ta]:
            return stance_class_to, stance_prob_to
        else:
            return stance_class_ta, stance_prob_ta
