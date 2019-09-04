import os
import json
import sys
from joblib import load
import numpy as np

from .features.extract_features import Features
from .util import Util, path_from_root

RESOURCES_PATH = path_from_root("resources.txt")

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
        """
        :param source: JSON dict object source
        :param reply: JSON dict object reply
        :return: stance_class, stance_prob
            stance_class float
                0.0 = support
                1.0 = deny
                2.0 = query
                3.0 = comment
            stance_prob [support_prob, deny_prob, query_prob, comment_prob]
        """
        #Load resources:
        clf = load(path_from_root(self.resources["model_" + self.model]))
        tweet_features = np.array(self.feature_extractor.features(source, reply))
        stance_class = clf.predict(tweet_features.reshape(1, -1))[0]
        stance_prob = clf.predict_proba(tweet_features.reshape(1, -1))[0]
        return stance_class, stance_prob



