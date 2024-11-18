#!/usr/bin/env python3
#
#    Copyright 2020 European Language Grid
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#
# Very simple-minded whitespace tokeniser that returns a Token annotation for
# every sequence of non-whitespace characters in the supplied text.
#

from flask import Flask, request
from werkzeug.exceptions import BadRequest
import cgi
import codecs
import os

import collections
import json
import re
import warnings

from docker_classifier import oblivious_mode, classifier

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# Take plain text post size limit from env
size_limit = int(os.environ.get('REQUEST_SIZE_LIMIT', 50000))

class EmbeddedJSON(collections.namedtuple("EmbeddedTweet", "json begin end")):
    """
    This class is a loaded JSON object (at .json) associated
    with .begin and .end offsets within the string whence it was parsed.
    """

JSON = json.JSONDecoder()

class RequestTooLarge(Exception):
    pass

def invalid_request_error(e):
    """Generates a valid ELG "failure" response if the request cannot be parsed"""
    return {'failure':{ 'errors': [
        { 'code':'elg.request.invalid', 'text':'Invalid request message' }
    ] } }, 400

app.register_error_handler(BadRequest, invalid_request_error)
app.register_error_handler(UnicodeError, invalid_request_error)

@app.errorhandler(RequestTooLarge)
def request_too_large(e):
    """Generates a valid ELG "failure" response if the request is too large"""
    return {'failure':{ 'errors': [
        { 'code':'elg.request.too.large', 'text':'Request size too large' }
    ] } }, 400

#def entity_annotation(sp):
#    """Converts a spacy Span object into an ELG-compliant annotation.
#    Returns a tuple of the entity type and the ELG annotation object"""
#    return (sp.label_, {
#      'start':sp.start_char, 'end':sp.end_char,
#      'features':{'string':sp.text, 'id':sp.kb_id_, 'sentiment':sp.sentiment}
#    })

def get_text_content(request, charset):
    decoder_factory = codecs.getincrementaldecoder(charset)
    decoder = decoder_factory(errors='strict')
    content = ''
    limit_hit = False
    for chunk in request.body:
        if not(limit_hit):
            content += decoder.decode(chunk)
        if len(content) > size_limit:
            limit_hit = True
    if limit_hit:
        raise RequestTooLarge()

    content += decoder.decode(b'', True)
    return content

@app.route('/process', methods=['POST'])
def process_request():
    """Main request processing logic - accepts a JSON request and returns a JSON response."""
    ctype, type_params = cgi.parse_header(request.content_type)
    if ctype == 'text/plain':
        content = get_text_content(request, type_params.get('charset', 'utf-8'))
        return process(iter_json(content))
    elif ctype == 'application/json':
        data = request.get_json()
        # sanity checks on the request message
        if data.get('type') == 'text':
            if 'content' not in data:
                raise BadRequest()
            return process(iter_json(data['content']))
        elif data.get('type') == 'structuredText':
            # check that the request is a "flat" request, i.e. all the top
            # level texts are leaf nodes
            if 'texts' not in data or any('content' not in text for text in data['texts']):
                raise BadRequest()
            return process(structured_json(data['texts']))
        else:
            raise BadRequest()
    else:
        raise BadRequest()

def process(twt_source):
    annotations = dict()

    tweets_by_id = {}

    for tweet in twt_source:
        tweet_id = tweet.json.get("id_str")
        if tweet_id in tweets_by_id:
            warnings.warn("Tweet id {} has been duplicated".format(tweet_id))
        tweets_by_id[tweet_id] = tweet

    replies = filter_replies(tweets_by_id)

    for reply in replies:
        tweet_id = reply.json.get("id_str")
        in_reply_to = reply.json.get("in_reply_to_status_id_str")

        if oblivious_mode:
            classification, scores = classifier.classify(reply.json)
        else:
            original = tweets_by_id[in_reply_to]
            classification, scores = classifier.classify_with_target(reply.json, original.json)


        features = stance_classification_as_features(classification, scores)
        # Add ID and reply features, if they are not auto-generated IDs
        if not tweet_id.startswith('gen:'):
            features["tweet_id"] = tweet_id
        if not in_reply_to.startswith('gen:'):
            features["in_reply_to"] = in_reply_to

        annot = {"start": reply.begin, "end": reply.end, "features": features}
        annotations.setdefault("Stance", []).append(annot)

    return dict(response = { 'type':'annotations', 'annotations':annotations,  })

def filter_replies(tweets_by_id):
    """
    The input is a dict mapping tweet ID to EmbeddedJSON

    Fitlers the list to the replies, so that
    the return list consists of tweets
    that are replies to tweets that are in the input list.

    Note that this is a filter in the sense that all of the
    elements of the result list must have appeared in the input.
    """

    # All the tweets that are valid replies
    replies = []

    for tweet in tweets_by_id.values():
        reply_to = tweet.json.get("in_reply_to_status_id_str")
        if reply_to: # and reply_to in tweets_by_id:
            replies.append(tweet)

    return replies

def iter_json(text):
    """
    From a string input, iteratively parse JSON documents,
    until there are no more JSON documents to parse.
    This allows several JSON documents to be concatenated
    and then parsed one-by-one using this iterator.

    Each separate JSON document will be yielded by this function
    as an instance of the namedtuple "json begin end":
    .json is the JSON string parsed into Python objects;
    .begin index in text of the beginning of the JSON;
    .end index in text of the end of the JSON.

    JSON documents in the input can be separated from each other
    with whitespace, or nothing at all.

    If the text content does not start with a valid JSON object,
    then we assume that the whole text is instead the plain text
    content of a single reply, and synthesize a response
    accordingly.
    """

    p = 0
    found_json = False
    while True:
        # Skip initial whitespace,
        # because raw_decode _still_ doesn't handle it.
        # https://bugs.python.org/issue15393

        m = re.match(r"(?:\s|,|\[|\])*", text[p:])
        p += m.end()

        start_index = p

        # We only care about JSON _objects_, if there is another type of
        # JSON value starting here (number, string literal, boolean, null or
        # array) then give up.
        if text[p:p+1] != "{":
            break

        try:
            a_json, length = JSON.raw_decode(text[p:])
            found_json = True
        except json.decoder.JSONDecodeError:
            break

        # tweets.append(tw)
        p += length
        end_index = p

        yield EmbeddedJSON(a_json, start_index, end_index)

    if not found_json and oblivious_mode:
        trimmed_text = text.lstrip()
        text_start = len(text) - len(trimmed_text)
        trimmed_text = trimmed_text.rstrip()
        if not trimmed_text:
            return

        # We didn't find any JSON, but we did find some plain text,
        # so treat that as a single reply post to a dummy target
        yield EmbeddedJSON(
            {
                "text": trimmed_text,
                "id_str": "gen:1",
                "in_reply_to_status_id_str": "gen:0"
            },
            text_start,
            text_start + len(trimmed_text),
        )

def structured_json(texts_list):
    """
    Convert structured text JSON format into the iterator-of-twitter-jsons that
    the classifier expects.
    """
    # do any of the texts in our list have explicit id or reply_to features? If
    # not, we assume the first entry is an original and all the other entries
    # are replies to it.
    explicit_refs = any(("id" in f or "reply_to" in f) for f in (doc.get('features', {}) for doc in texts_list))

    for idx, doc in enumerate(texts_list):
        twt_json = {
            'text':doc['content'],
            'id_str':str(doc.get('features', {}).get('id', f'gen:{idx}')),
        }

        # Only link reply_to back to the first text if there are no explicit refs
        reply_to = doc.get('features', {}).get('reply_to', 'gen:0' if (idx > 0 and not explicit_refs) else None)
        if reply_to:
            twt_json['in_reply_to_status_id_str'] = str(reply_to)

        yield EmbeddedJSON(twt_json, idx, idx+1)

def stance_classification_as_features(class_index, scores):
    """
    Takes as input the integer class and scores array and
    returns a dictionary of features that can be used on a GATE
    Basic Document annotation.

    The input is described in the StanceClassifier README:
    https://github.com/GateNLP/StanceClassifier#usage
    """

    #print(class_index)
    #print(scores)

    class_index = int(class_index)

    result = dict()
    classes = ["support", "deny", "query", "comment"]

    for key, score in zip(classes, scores):
        result[key + "_score"] = score.item()

    result["stance_class"] = classes[class_index]

    return result

if __name__ == '__main__':
    app.run()
