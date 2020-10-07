import numpy as np
import ktrain

def predict_ensemble(clf_list, scl_list, tweet_features):
    predictions = []
    for i in range(len(clf_list)):
        clf = clf_list[i]
        scl = scl_list[i]
        Xtest_scl = scl.transform(tweet_features)
        predictions.append(clf.predict_proba(Xtest_scl))
	
    predictions = np.array(predictions).sum(axis = 0) / len(clf_list)
    
    final_pred = np.argmax(predictions, axis = 1)[0]
    final_prob = np.max(predictions, axis = 1)[0]
 
    return final_pred, final_prob

def predict_bert(clf, tweet_pair):
    prediction_prob = np.array(clf.predict(tweet_pair, return_proba = True)) 
    threshold = [0.198,0.078,0.077,0.645]
    predction = prediction_prob / threshold
    for_standardize = prediction.sum()
    final_pred = np.argmax(predction, axis = 1)
    final_prob = np.max(predction, axis = 1) / for_standardize
    return final_pred, final_prob