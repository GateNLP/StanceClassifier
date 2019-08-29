# support = 0
# deny = 1
# query = 2
# comment = 3

import numpy as np
import sys
import warnings
from sklearn.exceptions import ConvergenceWarning
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
from joblib import dump, load
from sklearn.metrics import confusion_matrix, precision_score, f1_score, accuracy_score, recall_score
import json
import glob
from StanceClassifier.training.preprocesstwitter import PreprocessTwitter
from StanceClassifier.features.extract_features import Features
from StanceClassifier.util import Util

### DEPRECATED ###
GLOVE_SIZE = 300
TWITTER_SIZE = 200
EMB_TWITTER = True
EMB_FILE = "/export/data/carolina/stanceclassification/resources/glove.840B.300d.txt"
EMB_TWITTER_FILE = "/export/data/carolina/stanceclassification/resources/glove.twitter.27B.200d.txt"
BINARY = False
### (END) DEPRECATED ###

def new_features(l_file, dataset, feature_extractor):
    print("Extracting features")
    featset = []
    labels = []
    ids = []
    for l in l_file.keys():
        r_json = dataset[l][0]
        st_id = dataset[l][1]
        s_json = dataset[st_id][0]

        ids.append(l)

        featset.append(np.array(feature_extractor.features(s_json, r_json)))
        
        if BINARY:
            labels.append(build_labels_bin(l_file, l))
        else:
            labels.append(build_labels(l_file, l))

    print("Done.")
    return np.array(featset), np.array(labels), ids


def open_json(f):
    with open(f) as json_file:
        data = json.load(json_file)
        return data

def new_load_folder(folder):
    folders = glob.glob(folder + "/*/*")
    dataset = {}
    print("Loading Folders")
    for f in folders:
        st_id = f.split("/")[-1]
        ct_id = st_id

        st_file = open(f + "/source-tweet/" + st_id + ".json")
        st_json = json.load(st_file)
        dataset[st_id] = []
        dataset[st_id].append(st_json)
        dataset[st_id].append(st_id)
        replies = glob.glob(f + "/replies/*.json")

        for r in replies:
            r_id = r.split("/")[-1].replace(".json","")
            r_file = open(r)
            r_json = json.load(r_file)
            dataset[r_id] = []
            dataset[r_id].append(r_json)
            dataset[r_id].append(st_id)
    print("Done.",len(dataset.keys())," files loaded!")
    return dataset


def build_labels(l_file, l):
    label = 3.0
    if l_file[l] == "support":
        label = 0.0
    elif l_file[l] == "deny":
        label = 1.0
    elif l_file[l] == "query":
        label = 2.0
    elif l_file[l] == "comment":
        label = 3.0
    return label
#### DEPRECATED FUNCTIONS #####
#### LEFT ONLY FOR REFERENCE ####
#### THEY WILL BE REMOVED IN THE NEXT VERSION ####

def load_folder(folder):
    folders = glob.glob(folder + "/*/*")
    dataset = {}
    tokenizer = PreprocessTwitter()
    print("Loading Folders")
    for f in folders:
        st_id = f.split("/")[-1]
        ct_id = st_id
        st_file = open(f + "/source-tweet/" + st_id + ".json")
        st_json = json.load(st_file)
        st_text = st_json["text"]
        dataset[st_id] = []
        dataset[st_id].append(tokenizer.tokenize(st_text))
        dataset[st_id].append(st_id)
        replies = glob.glob(f + "/replies/*.json")

        for r in replies:
            r_id = r.split("/")[-1].replace(".json","")
            r_file = open(r)
            r_json = json.load(r_file)
            r_text = r_json["text"]
            dataset[r_id] = []
            dataset[r_id].append(tokenizer.tokenize(r_text))
            dataset[r_id].append(st_id)
    print("Done.",len(dataset.keys())," files loaded!")
    return dataset


def tweet_vector(words, glove):
    size = GLOVE_SIZE
    if EMB_TWITTER:
        size = TWITTER_SIZE
    t_vector = np.zeros(size)
    for w in words:
        if w in glove.keys():
            t_vector = t_vector + np.array(glove[w])
    t_vector = t_vector/float(len(words))
    return t_vector



def build_labels_bin(l_file, l):
    label = 0.0
    if l_file[l] == "support":
        label = 1.0
    elif l_file[l] == "deny":
        label = 1.0
    elif l_file[l] == "query":
        label = 1.0
    elif l_file[l] == "comment":
        label = 0.0
    return label

def features(l_file, dataset, glove, hc, ids_hc):
    print("Extracting features")
    featset = []
    labels = []
    c = 0
    for l in ids_hc:
        r_words = dataset[l][0].split(" ")
        st_id = dataset[l][1]
        r_vector = tweet_vector(r_words, glove)

        st_words = dataset[st_id][0].split(" ")
        st_vector = tweet_vector(st_words, glove)

        #find_hc = np.where(ids_hc==l)
        hc_vector = hc[c]
        c = c + 1

        aux_vector = np.append(st_vector, r_vector)
        final_vector = np.append(aux_vector, hc_vector)

        featset.append(final_vector)
        if BINARY:
            labels.append(build_labels_bin(l_file, l))
        else:
            labels.append(build_labels(l_file, l))


    print("Done.")
    return np.array(featset), np.array(labels)
    
#### (END) DEPRECATED FUNCTIONS #####


def train(model, X, y, name):
    print("Training model: " + model)
    if model == "ens":
        clf1 = LogisticRegression(solver="saga", class_weight="balanced")
        clf2 = RandomForestClassifier()
        clf3 = MLPClassifier(max_iter=1000)
        eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('mlp', clf3)], voting='soft')
        params = {'lr__C': [1e0, 1e1, 1e2, 1e3],
                  'lr__tol': [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6],
                  'lr__multi_class': ["ovr", "multinomial"],
                  'lr__penalty': ["l1", "l2", "elasticnet"],
                  'lr__l1_ratio': [0.0, 0.0001, 0.001, 0.005, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0], 
                  'rf__n_estimators': [400, 600, 800],
                  'mlp__solver':["sgd", "adam", "lbfgs"],
                  'mlp__activation' : ["identity", "logistic", "tanh", "relu"],
                  'mlp__alpha': [0.0, 0.0001, 0.001, 0.005, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
                  'mlp__tol': [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6]
                  }
        clf = GridSearchCV(estimator=eclf, param_grid=params, cv=5, n_jobs=10)
    
    if model == "gp":
        clf = GaussianProcessClassifier(1.0 * RBF(1.0))

    if model == "mlp":
        clf = GridSearchCV(cv=5, estimator=MLPClassifier(alpha=1, max_iter=1000), param_grid = 
                    {'solver':["sgd", "adam", "lbfgs"],
                     'activation' : ["identity", "logistic", "tanh", "relu"],
                     'alpha': [0.0, 0.0001, 0.001, 0.005, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
                     'tol': [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6]
                    }, n_jobs=10)

    if model == "rf":
        clf = RandomForestClassifier(n_estimators=500)

    if model == "svm":
        clf = GridSearchCV(cv=5, estimator=SVC(kernel='rbf', gamma=0.1), param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "gamma": np.logspace(-2, 2, 5)}, n_jobs=10)

    if model == "lr":
        clf = GridSearchCV(cv=5, estimator=LogisticRegression(solver="saga", class_weight="balanced"), param_grid={
                                "tol":[1e-1,1e-2,1e-3,1e-4,1e-5,1e-6], 
                                "C": [1e0, 1e1, 1e2, 1e3], 
                                "multi_class": ["ovr", "multinomial"],
                                "penalty": ["l1", "l2", "elasticnet"], 
                                "l1_ratio": [0.0, 0.0001, 0.001, 0.005, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]}, n_jobs=10)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
        clf.fit(X,y)
    dump(clf, name+".model")
    print("Done.")
    return clf


def test(clf, X_test):
    return clf.predict(X_test)

def evaluation(y_test, y_pred, model):
    f_eval = open(model+".eval", "w")
    p_macro = precision_score(y_test, y_pred, average='macro')
    p_micro = precision_score(y_test, y_pred, average='micro')
    p_weighted = precision_score(y_test, y_pred, average='weighted')
    r_macro = recall_score(y_test, y_pred, average='macro')
    r_micro = recall_score(y_test, y_pred, average='micro')
    r_weighted = recall_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print("F1_micro: %f" % f1_micro)
    print("F1_macro: %f" % f1_macro)
    print("F1_weighted: %f" % f1_weighted)
    print("Accuracy: %f" % accuracy)
    print(cm)
    f_eval.write("Precision macro: " + str(p_macro) + "\nPrecision micro: " + str(p_micro) + "\nPrecision weighted: " + str(p_weighted)+"\n\n")
    f_eval.write("Recall macro: " + str(r_macro) + "\nRecall micro: " + str(r_micro) + "\nRecall weighted: " + str(r_weighted)+"\n\n")
    f_eval.write("F1 macro: " + str(f1_macro) + "\nF1 micro: " + str(f1_micro) + "\nF1 weighted: " + str(f1_weighted)+"\n\n")
    f_eval.write("Acurracy: " + str(accuracy) + "\n\n")
    f_eval.write("Confusion Matrix:\n" + str(cm) + "\n\n")
    f_eval.close()


def save_json(model, y_pred, ids):
    _class = {0.0: "support", 1.0: "deny", 2.0: "query", 3.0: "comment"}
    data = {"subtaskaenglish": {}}
    for i in range(0, len(ids)): 
        data["subtaskaenglish"][str(ids[i])] = _class[y_pred[i]]
    with open(model+'.json', 'w') as json_file:  
        json.dump(data, json_file)



#Load resources:
print("Loading resources")
util = Util()
resources = util.loadResources('resources.txt')
print("Done. %d resources added!" % len(resources.keys()))



train_dict = open_json("/export/data/carolina/rumoureval/2017/semeval2017-task8-dataset/traindev/rumoureval-subtaskA-train.json")
dev_dict = open_json("/export/data/carolina/rumoureval/2017/semeval2017-task8-dataset/traindev/rumoureval-subtaskA-dev.json")
test_dict = open_json("/export/data/carolina/rumoureval/2017/subtaskA/test_data/subtaskA.json")


#### DEPRECATED ####

#if EMB_TWITTER:
#    glove = loadGloveModel(EMB_TWITTER_FILE)
#else:
#    glove = loadGloveModel(EMB_FILE)

####load data
#X_hc = np.loadtxt("/export/data/carolina/stanceclassification/semEvalFeatures.csv", skiprows=1, usecols=(0, 97))
#X_test_hc = np.loadtxt("/export/data/carolina/stanceclassification/semEvalFeaturesTesting.csv", skiprows=1, usecols=(0, 97))


#ids_hc = np.loadtxt("/export/data/carolina/stanceclassification/semEvalFeatures.csv" , skiprows=1, usecols=(6426), dtype=str)
#ids_test_hc = np.loadtxt("/export/data/carolina/stanceclassification/semEvalFeaturesTesting.csv" , skiprows=1, usecols=(6426), dtype=str)

#### (END) DEPRECATED ####


dataset = new_load_folder("/export/data/carolina/rumoureval/2017/semeval2017-task8-dataset/rumoureval-data/")

feature_type = "emb_hc" 

feature_extractor = Features(resources)

X, y, ids = new_features(train_dict, dataset, feature_extractor)
X_test, y_test, ids_test = new_features(test_dict, dataset, feature_extractor)


#### DEPRECATED ####
#X, y = features(train_dict, dataset, glove, X_hc, ids_hc)
#X_test, y_test = features(test_dict, dataset, glove, X_test_hc, ids_test_hc)
#### (END) DEPRECATED ####


scaler = StandardScaler()
scaler.fit(X)

X_scl = scaler.transform(X)
#X_scl = X

X_test_scl = scaler.transform(X_test)
#X_test_scl = X_test


#choose classifier
model = "rf"
folder = "training/models/"
name = folder + model + "_" + feature_type + "_esize" + resources['embeddings_size']

#train model
clf = train(model, X_scl, y, name)

#predict
y_pred = test(clf, X_test_scl)

#evaluation
evaluation(y_test, y_pred, name)

#save in the RumourEval format
save_json(name, y_pred, ids_test)
