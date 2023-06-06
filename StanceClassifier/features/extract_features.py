# support = 0
# deny = 1
# query = 2
# comment = 3

import numpy as np
from joblib import dump, load
from transformers import AutoTokenizer
import json
import glob
import re
from StanceClassifier.util import Util, path_from_root
import emoji
from nltk.tokenize import TweetTokenizer
import string

class Features():

    def __init__(self, tokenizer_PATH):
    
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_PATH)
    
    def process_tweet_dict(self, tweet_dict):

        if "text" not in tweet_dict.keys():
            text = tweet_dict["full_text"]
        else:
            text = tweet_dict["text"]
        
        
        tknzr = TweetTokenizer()
        FLAGS = re.MULTILINE | re.DOTALL
        text = re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "HTTPURL", text, flags=FLAGS)
        text = re.sub(r"@\w+", "@USER", text, flags=FLAGS)
        text_token = tknzr.tokenize(text)
        text = " ".join(text_token)
        text = emoji.demojize(text)
        
        return text
    
    def extract_bert_input(self, reply_tweet_dict):
       
        r_text = self.process_tweet_dict(reply_tweet_dict)
        
        # input of target-oblivious model
        encoded_reply = self.tokenizer(text=r_text, add_special_tokens=True, truncation=True, padding='max_length', max_length = 128, return_tensors="pt")
        return encoded_reply