import os
import json
import sys
from joblib import load
import numpy as np
from .features.extract_features import Features
from .testing import test
from transformers import AutoModelForSequenceClassification,AutoTokenizer



class StanceClassifier:

    def __init__(self, model="GateNLP/stance-twitter-xlm-target-oblivious", tokenizer=None):
        if not tokenizer:
            tokenizer = model

        self.feature_extractor = Features(tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=4)
        
    
    def classify(self, reply): 	
        
        encoded_reply, _ = self.feature_extractor.extract_bert_input(reply)
        #print("stanceclassifier.classify....................", encoded_reply, encoded_source_reply)
        stance_class, stance_prob = test.predict_bertweet(encoded_reply, self.model)
			        
        return stance_class, stance_prob


class StanceClassifierEnsemble(StanceClassifier):
    """
    Ensemble classifier that runs a target-oblivious and a target-aware model against
    the same pair of posts and returns whichever prediction is more confident.
    """

    def __init__(self, to_model="GateNLP/stance-bertweet-target-oblivious", ta_model="GateNLP/stance-bertweet-target-aware", tokenizer=None):
        super().__init__(to_model, tokenizer)
        self.ta_model = AutoModelForSequenceClassification.from_pretrained(ta_model, num_labels=4)


    def classify_with_target(self, reply, target):
        encoded_reply, reply_text = self.feature_extractor.extract_bert_input(reply)
        encoded_reply_and_target, _ = self.feature_extractor.extract_bert_input(target, reply_text)

        # run both the target oblivious and the target aware model, and return whichever gives
        # the higher score
        stance_class_to, stance_prob_to = test.predict_bertweet(encoded_reply, self.model)
        stance_class_ta, stance_prob_ta = test.predict_bertweet(encoded_reply_and_target, self.ta_model)
        if stance_prob_to[stance_class_to] > stance_prob_ta[stance_class_ta]:
            return stance_class_to, stance_prob_to
        else:
            return stance_class_ta, stance_prob_ta
