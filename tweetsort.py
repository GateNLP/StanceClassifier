#!/usr/bin/env python

"""
Take tweets as input.
Sort into "original" and "reply" tweets.
"""

import collections
import json
import re
import sys
import warnings

JSON = json.JSONDecoder()

def parse_tweets(inp):
    """
    Successively parse complete JSON objects from the entire input.
    Return a list of those objects.
    """
    text = inp.read()
    tweets = []

    while True:

        # Trim initial whitespace,
        # because raw_decode _still_ doesn't handle it.
        # https://bugs.python.org/issue15393
        text = re.sub(r"\s*", "", text)
        if not text:
            break

        try:
            tw,length = JSON.raw_decode(text)
        except json.decoder.JSONDecodeError:
            break

        tweets.append(tw)
        text = text[length:]

    return tweets


def read_replies(inp):
    """
    Returns a pair of:
        (tweets, replies)
    Both dicts, indexed by the id_str of the original tweet.
    """
    tweets = parse_tweets(inp)

    # All the tweets, indexed by their id_str
    tweet_dict = dict()
    # All the replies to a tweet (as a list), indexed by the
    # id_str of the original tweet (to which the replies are replying).
    reply_dict = collections.defaultdict(list)

    for tweet in tweets:
        tweet_id = tweet['id_str']
        if tweet_id in tweet_dict:
            warnings.warn("Tweet id {} has been duplicated".format(tweet_id))
            continue
        tweet_dict[tweet_id] = tweet

        reply_to = tweet.get('in_reply_to_status_id_str')
        if reply_to:
            reply_dict[reply_to].append(tweet)

    return tweet_dict, reply_dict


def main(argv=None):
    if argv is None:
        argv = sys.argv
  
    ts, rs, = read_replies(sys.stdin)
    for original_id, replies in rs.items():
        tick = original_id in ts
        for reply in replies:
            print(reply['id_str'], "replying to", original_id, "✓" if tick else "?")

if __name__ == "__main__":
    main()
