import os
import json
import sys
from joblib import load
import numpy as np

from .features.extract_features import Features
from .util import *

RESOURCES_PATH = os.path.join(get_project_root(), "resources.txt")

class StanceClassifier():

    def __init__(self, model):

        self.model = model
        #Load resources:
        print("Loading resources")
        util = Util()

        self.resources = util.loadResources(RESOURCES_PATH)
        print("Done. %d resources added!" % len(self.resources.keys()))
        self.feature_extractor = Features(self.resources)

    def classify(self, source, reply): 
        #Load resources:
        clf = load(path_from_root(self.resources["model_" + self.model]))
        tweet_features = np.array(self.feature_extractor.features(source, reply))
        stance_class = clf.predict(tweet_features.reshape(1, -1))[0]
        stance_prob = clf.predict_proba(tweet_features.reshape(1, -1))[0]
        return stance_class, stance_prob



