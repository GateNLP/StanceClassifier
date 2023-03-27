import sys
import argparse
import json
from StanceClassifier.stance_classifier import StanceClassifier
import time



#Load config: 
#configurations = loadResources('../configurations.txt')

parser = argparse.ArgumentParser(description='Stance classifier for the WeVerify project.')
#parser.add_argument('-l', help='language', choices=['en','multilingual'])
parser.add_argument('-s', help='stance file to be classified (json file)')
parser.add_argument('-o', help='original stance file (json file)')
#parser.add_argument('-c', help='classifier', default = 'bert-multilingual', choices=['bert-multilingual'])


args = parser.parse_args()

v_args = vars(args)
start_time = time.time()
stance = json.load(open(v_args['s'], "r"))
original = json.load(open(v_args['o'], "r"))

classifier = StanceClassifier()
print("--- %s seconds ---" % (time.time() - start_time))

print(classifier.classify(original, stance)) # result
print("--- %s seconds ---" % (time.time() - start_time))
