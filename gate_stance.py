# This is a Python script that is intended to be run
# in GATE using the GATE plugin Python:
# http://gatenlp.github.io/gateplugin-Python/

# https://docs.python.org/3.6/library/collections.html
import collections
import json
import re
import warnings

from gatenlp import GateNlpPr, interact


JSON = json.JSONDecoder()


class EmbeddedJSON(collections.namedtuple("EmbeddedTweet", "json begin end")):
    pass


@GateNlpPr
def run(doc, **kwargs):
    text = doc.text

    annotations = doc.get_annotations()

    # Divide into valid JSON documents, and annotate

    tweets_by_id = {}
    for tweet in iter_json(text):
        tweet_id = tweet.json.get("id_str")
        if tweet_id in tweets_by_id:
            warnings.warn("Tweet id {} has been duplicated".format(tweet_id))
        tweets_by_id[tweet_id] = tweet

    replies = filter_replies([tw.json for tw in tweets_by_id.values()])

    for reply in replies:
        embedded = tweets_by_id[reply.get("id_str")]
        tweet_id = embedded.json.get("id_str")
        in_reply_to = embedded.json.get("in_reply_to_status_id_str")
        annotations.add(
            embedded.begin,
            embedded.end,
            "Tweet JSON",
            {"tweet_id": tweet_id, "in_reply_to": in_reply_to},
        )


def filter_replies(tweets):
    """
    The input is a list of tweets.

    Fitlers the list to the replies, so that
    the return list consists of tweets
    that are replies to tweets that are in the input list.

    Note that this is a filter in the sense that all of the
    elements of the result list must have appeared in the input.
    """

    # All the tweets, indexed by their id_str
    tweet_dict = dict()
    # All the replies to a tweet (as a list), indexed by the
    # id_str of the original tweet (to which the replies are replying).
    replies = []

    for tweet in tweets:
        tweet_id = tweet["id_str"]
        if tweet_id in tweet_dict:
            warnings.warn("Tweet id {} has been duplicated".format(tweet_id))
            continue
        tweet_dict[tweet_id] = tweet

        reply_to = tweet.get("in_reply_to_status_id_str")
        if reply_to and reply_to in tweet_dict:
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

    JSON documents in the input can be separate from each other
    with whitespace, or nothing at all.
    """

    p = 0
    while True:
        # Skip initial whitespace,
        # because raw_decode _still_ doesn't handle it.
        # https://bugs.python.org/issue15393

        m = re.match(r"\s*", text[p:])
        p += m.end()

        start_index = p

        try:
            a_json, length = JSON.raw_decode(text[p:])
        except json.decoder.JSONDecodeError:
            break

        # tweets.append(tw)
        p += length
        end_index = p

        yield EmbeddedJSON(a_json, start_index, end_index)


interact()
