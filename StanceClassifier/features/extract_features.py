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
    
        self.tokenizer_bertweet = AutoTokenizer.from_pretrained(tokenizer_PATH, use_fast=False)
    
    def process_tweet_dict_for_bertweet(self, tweet_dict):

        if "text" not in tweet_dict.keys():
            text = tweet_dict["full_text"]
        else:
            text = tweet_dict["text"]
        #print("text is................", text)
        
        tknzr = TweetTokenizer()
        FLAGS = re.MULTILINE | re.DOTALL
        
        text = re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "HTTPURL", text, flags=FLAGS)
        #print(text)
        text = re.sub(r"@\w+", "@USER", text, flags=FLAGS)
        #print(text)
        text_token = tknzr.tokenize(text)
        #print(text_token)
        text = " ".join(text_token)
        #print(text)
        
        text = emoji.demojize(text)
        #print(text)
        return text
    
    def extract_bert_input(self, source_tweet_dict, reply_tweet_dict):
       
        s_text = self.process_tweet_dict_for_bertweet(source_tweet_dict) 
        #print("s_text is...............",s_text)
        r_text = self.process_tweet_dict_for_bertweet(reply_tweet_dict)
        #print("r_text is...............",r_text)

        # input of target-oblivious model
        encoded_reply = self.tokenizer_bertweet(text=r_text, add_special_tokens=True, truncation=True, padding='max_length', max_length = 128, return_tensors="pt")
        #print("encoded_reply is.................", encoded_reply)
        # input of target-aware model
        encoded_source_reply = self.tokenizer_bertweet(text=s_text, text_pair = r_text, add_special_tokens=True, truncation=True, padding='max_length', max_length = 128, return_tensors="pt")
        #print("encoded_source_reply is..................", encoded_source_reply)
        return encoded_reply, encoded_source_reply