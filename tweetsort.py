#!/usr/bin/env python

"""
Take tweets as input.
Sort into "original" and "reply" tweets.
"""

import json
import re
import sys

JSON = json.JSONDecoder()

def tweets(inp):
    text = inp.read()
    tweets = []

    while True:
        # successively parse complete JSON objects
        # from the entire input.
        print(len(tweets))

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

    for tweet in tweets:
        replyto = tweet.get('in_reply_to_status_id_str')
        print(tweet['id_str'], replyto)

def main(argv=None):
    if argv is None:
        argv = sys.argv
  
    tweets(sys.stdin)

if __name__ == "__main__":
    main()
