import os
import json
import sys
from joblib import load
import numpy as np
from .features.extract_features import Features
from .util import Util, path_from_root
from .testing import test
from transformers import AutoModelForSequenceClassification,AutoTokenizer



class StanceClassifier():

    def __init__(self):
        
        RESOURCES_PATH = path_from_root("resources.txt")

        print("Loading resources")
        util = Util()
        self.resources = util.loadResources(RESOURCES_PATH)
        self.feature_extractor = Features(path_from_root(self.resources["model_BERTweet_tokenizer"])) 
        self.model_TO = AutoModelForSequenceClassification.from_pretrained(path_from_root(self.resources["model_BERTweet_TO"]), num_labels=4)
        self.model_TA = AutoModelForSequenceClassification.from_pretrained(path_from_root(self.resources["model_BERTweet_TA"]), num_labels=4)
        
    
    def classify(self, source, reply): 	
        
        encoded_reply, encoded_source_reply = self.feature_extractor.extract_bert_input(source, reply)
        #print("stanceclassifier.classify....................", encoded_reply, encoded_source_reply)
        stance_class, stance_prob = test.predict_bertweet(encoded_reply, encoded_source_reply, self.model_TO, self.model_TA)
			        
        return stance_class, stance_prob