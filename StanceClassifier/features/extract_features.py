# support = 0
# deny = 1
# query = 2
# comment = 3

import numpy as np
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from scipy.spatial.distance import cosine
from joblib import dump, load
from sklearn.metrics import confusion_matrix, precision_score, f1_score, accuracy_score, recall_score
import json
import glob
from .preprocesstwitter import PreprocessTwitter
from .heuristics import Heuristics
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from StanceClassifier.util import Util, path_from_root


class Features():

    def __init__(self, resources): 
        self.emb_file = resources["embeddings_file"]
        self.emb_size = int(resources["embeddings_size"])
        self.emoticons = self.read_emoticon(resources["emoticon"])
        self.emoticons_cat = np.zeros(42)
        self.acronyms = self.read_acronyms(resources["acronyms"])
        self.vulgarWords = self.read_vulgar_words (resources["vulgar_words"])
        self.googleBadWords = self.read_google_bad_words(resources["google_bad_words"])
        self.surpriseWords = self.read_surprise_words(resources["affect_surprise"])
        self.doubtWords = self.read_doubt_words(resources["doubt_words"])
        self.noDoubtWords = self.read_no_doubt_words(resources["no_doubt_words"])

        self.scaler = load(path_from_root(resources["scaler"]))

        self.glove = self.loadGloveModel(self.emb_file)

        self.support_terms = ["support", "join", "confirm", "aid", "help"]


    def read_doubt_words(self, f):
        f_doubt = open(path_from_root(f), "r")
        v_doubt = []
        for l in f_doubt.readlines():
            tokens = l.split(",")
            for t in tokens:
                v_doubt.append(t.strip())
        return v_doubt

    def read_no_doubt_words(self, f):
        f_no_doubt = open(path_from_root(f), "r")
        v_no_doubt = []
        for l in f_no_doubt.readlines():
            tokens = l.split(",")
            for t in tokens:
                v_no_doubt.append(t.strip())
        return v_no_doubt

    def read_surprise_words(self, f):
        f_surprise = open(path_from_root(f), "r")
        v_surprise = []
        for l in f_surprise.readlines():
            t = l.split("\t")
            v_surprise.append(t[0].strip())
        return v_surprise

    def read_google_bad_words(self, f):
        f_google_bad = open(path_from_root(f), "r")
        v_google_bad = []
        for l in f_google_bad.readlines():
            t = l.split(":")
            v_google_bad.append(t[0].replace("\"", "").strip())
        return v_google_bad

    def read_vulgar_words(self, f):
        f_vulgar = open(path_from_root(f), "r")
        v_vulgar = []
        for l in f_vulgar.readlines():
            t = l.split("-")
            v_vulgar.append(t[0].strip())
        return v_vulgar


    def read_acronyms(self, f):
        f_acronym = open(path_from_root(f), "r")
        v_acronym = []
        for l in f_acronym.readlines():
            v_acronym.append(l.strip())
        return v_acronym

    def read_emoticon(self, f):
        f_emoticon = open(path_from_root(f), "r")
        dict_emoticon = {}
        for l in f_emoticon.readlines():
            t = l.strip().split("\t")
            dict_emoticon[t[0]] = int(t[1])
        return dict_emoticon



    def loadGloveModel(self, gloveFile):
        print("Loading Glove Model")
        f = open(path_from_root(gloveFile), 'r')
        model = {}
        for line in f:
            splitLine = line.split(" ")
            #print(splitLine)
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        print("Done.",len(model)," words loaded!")
        return model

    def open_json(self, f):
        with open(path_from_root(f)) as json_file:
            data = json.load(json_file)
            return data


    def tweet_vector(self, words):
        size = self.emb_size
        t_vector = np.zeros(size)
        for w in words:
            if w in self.glove.keys():
                t_vector = t_vector + np.array(self.glove[w])
        t_vector = t_vector/float(len(words))
        return t_vector

    def build_labels(self, l_file, l):
        labels = []
        if l_file[l] == "support":
            labels.append(0.0)
        elif l_file[l] == "deny":
            labels.append(1.0)
        elif l_file[l] == "query":
            labels.append(2.0)
        elif l_file[l] == "comment":
            labels.append(3.0)
        return labels


    def features(self, source, reply):
        tokenizer = PreprocessTwitter()
        featset = []
        hc = []
        c = 0


        if "text" not in source.keys():
            s_text_raw = source["full_text"].lower()
        else:
            s_text_raw = source["text"].lower()

        if "text" not in reply.keys():
            r_text_raw = reply["full_text"].lower()
        else:
            r_text_raw = reply["text"].lower()

        s_text = tokenizer.tokenize(s_text_raw)
        r_text = tokenizer.tokenize(r_text_raw)

        s_id = source["id"]
        r_id = reply["id"]

        s_words = s_text.split(" ")
        r_words = r_text.split(" ")

        s_vector = self.tweet_vector(s_words)
        r_vector = self.tweet_vector(r_words)


        #similarity between source and reply
        sourceSim = cosine(s_vector, r_vector)
        hc.append(sourceSim)


        ###has url? 
        url = 0.
        if "<url>" in r_text:
            url = 1.
        hc.append(url)

        ###ends with question mark
        ewqm = 0.
        if r_text[-1] == "?":
            ewqm = 1.
        hc.append(ewqm)

        ###is reply?
        is_reply = 0.
        if s_id != r_id:
            is_reply = 1.
        hc.append(is_reply)

        ###supporting similarity
        sup_vector = self.tweet_vector(self.support_terms)
        supSim = cosine(r_vector, sup_vector)
        hc.append(supSim)

        ###contains Wh question
        wh = 0.
        if "who" in r_words or "where" in r_words or "why" in r_words or "what" in r_words:
            wh = 1.
        hc.append(wh)

        ###dontyou
        dontyou = 0.
        if "don't you" in r_text:
            dontyou = 1.
        hc.append(dontyou)

        ###arentyou
        arentyou = 0.
        if "aren't you" in r_text:
            arentyou = 1.
        hc.append(arentyou)

        ###has replies
        has_replies = 0.
        if reply["in_reply_to_status_id"] != "null":
            has_replies = 1.
        hc.append(has_replies)

        ###sentiment features (Vader)
        #analyser = SentimentIntensityAnalyzer()
        #score = analyser.polarity_scores(r_text_raw)
        #hc.append(score["neg"])
        #hc.append(score["pos"])
        #hc.append(score["neu"])
        #hc.append(score["compound"])
        
        #score = analyser.polarity_scores(s_text_raw)
        #hc.append(score["neg"])
        #hc.append(score["pos"])
        #hc.append(score["neu"])
        #hc.append(score["compound"])

        sentAnalyser = SentimentIntensityAnalyzer()

        scores = sentAnalyser.polarity_scores(r_text_raw)

        hc.append(scores["neg"])
        hc.append(scores["pos"])
        hc.append(scores["neu"])
        hc.append(scores["compound"])

        scores = sentAnalyser.polarity_scores(s_text_raw)

        hc.append(scores["neg"])
        hc.append(scores["pos"])
        hc.append(scores["neu"])
        hc.append(scores["compound"])
        
        ###TODO: check negation features -- Stanford

        ###has negation

        ###average negation



        ###has slang/curse word
        hasVulgar = 0.

        ###has Google bad word
        hasGoogleBadWords = 0.

        ###has acronyms
        hasAcro = 0.

        ###average word length
        wordCount = 0.
        wordLen = 0.
        avWL = 0.

        for token in r_words:
            if token[0] != "<" and token[-1] != ">":
                wordLen = wordLen + float(len(token))
                wordCount = wordCount + 1.
            if token in self.acronyms:
                hasAcro = 1.
            if token in self.vulgarWords:
                hasVulgar = 1.
            if token in self.googleBadWords:
                hasGoogleBadWords = 1.
        if wordCount != 0.:
            avWL = wordLen/wordCount
        
        hc.append(avWL)
        hc.append(hasAcro)
        hc.append(hasVulgar)
        hc.append(hasGoogleBadWords)

        ###surprise score
        surprise_vector = self.tweet_vector(self.surpriseWords)
        surprise_score = cosine(r_vector, surprise_vector)
        hc.append(surprise_score)

        ###doubt Score
        doubt_vector = self.tweet_vector(self.doubtWords)
        doubt_score = cosine(r_vector, doubt_vector)
        hc.append(doubt_score)
        
        ###nodoubt score
        no_doubt_vector = self.tweet_vector(self.noDoubtWords)
        no_doubt_score = cosine(r_vector, no_doubt_vector)
        hc.append(no_doubt_score)

        ###has question mark
        ###number question mark
        numberQM = r_text_raw.count("?")
        hasQM = 0.
        if numberQM > 0:
            hasQM = 1.
        hc.append(hasQM)
        hc.append(float(numberQM))

        ###has exclamation mark
        ###number exclamation mark
        numberEM = r_text_raw.count("!")
        hasEM = 0.
        if numberEM > 0:
            hasEM = 1.
        hc.append(hasEM)
        hc.append(float(numberEM))

        ###has dot dot dot
        ###number dot dot dot
        numberDDD = r_text_raw.count("...")
        hasDDD = 0.
        if numberDDD > 0:
            hasDDD = 1.
        hc.append(hasDDD)
        hc.append(float(numberDDD))

        ###originality (tweets counts)
        tweet_count = float(reply["user"]["statuses_count"])
        hc.append(tweet_count)

        ###is Verified
        isVerified = 0.
        if reply["user"]["verified"] != "false":
            isVerified = 1.
        hc.append(isVerified)

        ###number of followers
        followers = float(reply["user"]["friends_count"])
        hc.append(followers)

        ###role
        followees = float(reply["user"]["followers_count"])
        role = 0.
        if followees != 0.:
            role = followers/followees
        hc.append(role)

        #TODO: understand engagement features
        #import time
        #res = time.strptime(reply["user"]["created_at"], "%a %b %d %H:%M:%S +0000 %Y")
        #print(res)

        ###engagement

        ###engagement favorite


        ###public list membership count
        public_list = float(reply["user"]["listed_count"])
        hc.append(public_list)

        ###has geo enabled
        geo = 0.
        if reply["user"]["geo_enabled"] != "false":
            geo = 1.
        hc.append(geo)


        ###has description
        description = reply["user"]["description"]
        hasDesc = 0.
        len_desc = 0.
        if (description != None) and (description.strip() != ""):
            hasDesc = 1.
            len_desc = float(len(description.split(" ")))
        hc.append(hasDesc)


        ###length of description
        hc.append(len_desc)

        ###pattern 1
        pattern1 = Heuristics.pattern1(r_text_raw)
        hc.append(pattern1)
        
        ###pattern 2
        pattern2 = Heuristics.pattern2(r_text_raw)
        hc.append(pattern2)

        ###pattern 3
        pattern3 = Heuristics.pattern3(r_text_raw)
        hc.append(pattern3)

        ###pattern 4
        pattern4 = Heuristics.pattern4(r_text_raw)
        hc.append(pattern4)

        ###pattern 5
        pattern5 = Heuristics.pattern5(r_text_raw)
        hc.append(pattern5)

        ###pattern 6
        pattern6 = Heuristics.pattern6(r_text_raw)
        hc.append(pattern6)

        ###pattern 7
        pattern7 = Heuristics.pattern7(r_text_raw)
        hc.append(pattern7)

        ###pattern 8
        pattern8 = Heuristics.pattern8(r_text_raw)
        hc.append(pattern8)

        ###pattern 9
        pattern9 = Heuristics.pattern9(r_text_raw)
        hc.append(pattern9)

        ###pattern 10
        pattern10 = Heuristics.pattern10(r_text_raw)
        hc.append(pattern10)
        
        ###emoticons
        r_text_noURL = re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>", r_text_raw, flags=re.MULTILINE | re.DOTALL)

        for emo in self.emoticons.keys():
            if emo in r_text_noURL:
                self.emoticons_cat[self.emoticons[emo]] = 1.

        for emo_cat in self.emoticons_cat:
            hc.append(emo_cat)


        hc_vector = np.array(hc)


        aux_vector = np.append(s_vector, r_vector)
        final_vector = np.append(aux_vector, hc_vector)
        
        final_vector_scaler = self.scaler.transform(final_vector.reshape(1, -1))

        return final_vector_scaler
    

