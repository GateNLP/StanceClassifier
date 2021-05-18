import os
import json
import sys
from joblib import load
import numpy as np
import ktrain
from .features.extract_features import Features
from .util import Util, path_from_root
from .testing import test
RESOURCES_PATH = path_from_root("resources.txt")

class StanceClassifier():

    def __init__(self, model):

        self.model = model
        #Load resources:
        print("Loading resources")
        util = Util()
        
        self.resources = util.loadResources(RESOURCES_PATH)
        print("Done. %d resources added!" % len(self.resources.keys()))
		
        if (model == "bert-english") or (model == "bert-multilingual") or (model == "cloud-docker"):
            self.feature_extractor = Features(self.resources, only_text = True)
        else:
            self.feature_extractor = Features(self.resources, only_text = False)

        self.clf = None

        if model == "cloud-docker":
            self.clf = ktrain.load_predictor(path_from_root(self.resources["model_cloud_docker"]))
        
        if model == "ens":
            self.scaler_list = load(path_from_root(self.resources["scaler_ensemble"]))
            self.clf_list = load(path_from_root(self.resources["model_ensemble"]))
			
        if model == "bert-english":
            self.clf = ktrain.load_predictor(path_from_root(self.resources["model_bert_english"]))

        if model == "bert-multilingual":
            self.clf = ktrain.load_predictor(path_from_root(self.resources["model_bert_multilingual"]))
    
    def classify(self, source, reply):

        if self.clf is None:
            # Ensemble model
            tweet_features = np.array(self.feature_extractor.features(source, reply)).reshape(1, -1)
            stance_class, stance_prob = test.predict_ensemble(self.clf_list, self.scaler_list, tweet_features)
        else:
            # BERT model
            tweet_pair = self.feature_extractor.extract_text(source, reply)
            stance_class, stance_prob = test.predict_bert(self.clf, tweet_pair)
        
        return stance_class, stance_prob
