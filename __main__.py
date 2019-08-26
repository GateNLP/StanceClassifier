import sys
import argparse
import json
from Stance_Classifier import StanceClassifier



#Load config: 
#configurations = loadResources('../configurations.txt')

parser = argparse.ArgumentParser(description='Stance classifier for the WeVerify project.')
parser.add_argument('-l', help='language', choices=['en'])
parser.add_argument('-s', help='stance file to be classified (json file)')
parser.add_argument('-o', help='original stance file (json file)')
parser.add_argument('-c', help='classifier', choices=['lr', 'rf', 'svm', 'mlp'])


args = parser.parse_args()

v_args = vars(args)

stance = json.load(open(v_args['s'], "r"))
original = json.load(open(v_args['o'], "r"))

classifier = StanceClassifier(v_args['c'])

if v_args['l'] == 'en':
    print(classifier.classify(original, stance))
    



