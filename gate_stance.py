# This is a Python script that is intended to be run
# in GATE using the GATE plugin Python:
# http://gatenlp.github.io/gateplugin-Python/
import json
import re

from gatenlp import GateNlpPr, interact


JSON = json.JSONDecoder()


@GateNlpPr
def run(doc, **kwargs):
    text = doc.text

    annotations = doc.get_annotations()

    # Divide into valid JSON documents, and annotate

    p = 0
    while True:    
    
        # Skip initial whitespace,    
        # because raw_decode _still_ doesn't handle it.    
        # https://bugs.python.org/issue15393    

        m = re.match(r"\s*", text[p:])
        p += m.end()

        tweet_start = p
    
        try:    
            tw,length = JSON.raw_decode(text[p:])    
        except json.decoder.JSONDecodeError:    
            break    
    
        # tweets.append(tw)    
        p += length
        tweet_end = p

        annotations.add(tweet_start, tweet_end, "Tweet JSON",
          {"tweet_id": tw.get("id_str")})


interact()

