import sys
import argparse
import json
from StanceClassifier.stance_classifier import StanceClassifier
import time



#Load config: 
#configurations = loadResources('../configurations.txt')

parser = argparse.ArgumentParser(description='Multilingual Reply Only Stance classifier.')
parser.add_argument('-s', help='stance file to be classified (json file)')



args = parser.parse_args()

v_args = vars(args)
start_time = time.time()
stance = json.load(open(v_args['s'], "r"))

classifier = StanceClassifier()
print("--- %s seconds ---" % (time.time() - start_time))

print(classifier.classify(stance)) # result
print("--- %s seconds ---" % (time.time() - start_time))
