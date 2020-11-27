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
    
    def classify(self, source, reply):

        if self.model == "cloud-docker":
            tweet_pair = self.feature_extractor.extract_text(source, reply)
            clf = ktrain.load_predictor(path_from_root(self.resources["model_cloud_docker"]))
            stance_class, stance_prob = test.predict_bert(clf, tweet_pair)
        
        if self.model == "ens":
            tweet_features = np.array(self.feature_extractor.features(source, reply)).reshape(1, -1)
            scaler_list = load(path_from_root(self.resources["scaler_ensemble"]))
            clf_list = load(path_from_root(self.resources["model_ensemble"]))
            stance_class, stance_prob = test.predict_ensemble(clf_list, scaler_list, tweet_features)
			
        if self.model == "bert-english":
            tweet_pair = self.feature_extractor.extract_text(source, reply)
            clf = ktrain.load_predictor(path_from_root(self.resources["model_bert_english"]))
            stance_class, stance_prob = test.predict_bert(clf, tweet_pair)

        if self.model == "bert-multilingual":
            tweet_pair = self.feature_extractor.extract_text(source, reply)
            clf = ktrain.load_predictor(path_from_root(self.resources["model_bert_multilingual"]))
            stance_class, stance_prob = test.predict_bert(clf, tweet_pair)

			        
        #stance_class = clf.predict(tweet_features.reshape(1, -1))[0]
        #stance_prob = clf.predict_proba(tweet_features.reshape(1, -1))[0]
        return stance_class, stance_prob
