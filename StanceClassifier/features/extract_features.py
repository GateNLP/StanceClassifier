# support = 0
# deny = 1
# query = 2
# comment = 3
import functools

import numpy as np
from transformers import AutoTokenizer
import json
import glob
import re
import emoji
from nltk.tokenize import TweetTokenizer
import string

class Features:

    def __init__(self, tokenizer_PATH, tokenizer_kwargs=None, demojize=False):
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}
    
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_PATH, **tokenizer_kwargs)
        # Wrap the tokenizer in an LRU cache so we don't need to re-tokenize the reply text
        # twice when running in ensemble mode
        self.tokenizer = functools.lru_cache(maxsize=10)(self.tokenizer)
        self.demojize = demojize
        self.nltk_tokenizer = TweetTokenizer()


    def process_tweet_dict(self, tweet_dict):

        if "text" not in tweet_dict.keys():
            text = tweet_dict["full_text"]
        else:
            text = tweet_dict["text"]
        
        
        FLAGS = re.MULTILINE | re.DOTALL
        text = re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "http", text, flags=FLAGS)
        text = re.sub(r"@\w+", "@user", text, flags=FLAGS)
        text_token = self.nltk_tokenizer.tokenize(text)
        text = " ".join(text_token)

        if self.demojize:
            text = emoji.demojize(text)
        
        return text
    
    def extract_bert_input(self, tweet_dict, text_pair=None):
        """
        Preprocess and tokenize the given tweet dict ready for the stance model.

        :param tweet_dict: the tweet to process
        :param text_pair: optional supplementary text to send to the tokenizer.  Typically this
                will be omitted in the target-oblivious case, or when encoding just the reply tweet,
                or it will be the text returned when preprocessing the reply, when preparing the
                original tweet (the target to which it is replying) in a target-aware scenario.
        :return: tuple with the encoded result of this tweet ready to pass to the model, and the
                preprocessed text that could be passed as text_pair to encode the next tweet in the chain.
        """
        text = self.process_tweet_dict(tweet_dict)

        # input of target-oblivious model
        encoded = self.tokenizer(text=text, text_pair=text_pair, add_special_tokens=True, truncation=True, padding='max_length', max_length = 128, return_tensors="pt")
        return encoded, text