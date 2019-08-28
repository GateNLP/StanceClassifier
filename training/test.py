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
from preprocesstwitter import PreprocessTwitter
sys.path[0:0] = ["features/"]
from extract_features import Features
sys.path[0:0] = ["util/"]
from util import Util


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







train_dict = open_json("/export/data/carolina/rumoureval/2017/semeval2017-task8-dataset/traindev/rumoureval-subtaskA-train.json")
test_dict = open_json("/export/data/carolina/rumoureval/2019/eval-key-taskA.json")

dataset_train = new_load_folder("/export/data/carolina/rumoureval/2017/semeval2017-task8-dataset/rumoureval-data/")
dataset_test = new_load_folder("/export/data/carolina/rumoureval/2019/rumoureval-2019-test-data/twitter-en-test-data/")



print(dataset_train)

#Load resources:
print("Loading resources")
util = Util()
resources = util.loadResources('resources.txt')
print("Done. %d resources added!" % len(resources.keys()))

feature_type = "emb_hc" 

feature_extractor = Features(resources)

X, y, ids = new_features(train_dict, dataset_train, feature_extractor)
X_test, y_test, ids_test = new_features(test_dict, dataset_test, feature_extractor)

scaler = StandardScaler()
scaler.fit(X)


X_scl = scaler.transform(X)
#X_scl = X

dump(scaler, "models/scaler")

X_test_scl = scaler.transform(X_test)
#X_test_scl = X_test


#choose classifier
model = "lr"
folder = "models/"
name = folder + model + ".model"

clf = load(name)

#predict
y_pred = test(clf, X_test_scl)

#evaluation
evaluation(y_test, y_pred, name)

#save in the RumourEval format
save_json(name, y_pred, ids_test)
