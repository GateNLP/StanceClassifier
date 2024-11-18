import argparse
import json
from StanceClassifier import stance_classifier
import time



#Load config: 
#configurations = loadResources('../configurations.txt')

parser = argparse.ArgumentParser(description='Stance classifier.', epilog='If only a reply is provided, the classifier will use the multilingual target-oblivious model.  If both a reply and a target are provided then the classifier will use the target-aware ensemble model, which is currently English-only.')
parser.add_argument('reply', type=argparse.FileType('r'), help='JSON file with the reply tweet')
parser.add_argument('target', type=argparse.FileType('r'), nargs='?', help='JSON file with the target tweet (optional)')


args = parser.parse_args()

start_time = time.time()
reply = json.load(args.reply)

if args.target:
    classifier = stance_classifier.StanceClassifierEnsemble()
    target = json.load(args.target)

    print("--- %s seconds ---" % (time.time() - start_time))
    print(classifier.classify_with_target(reply, target))
else:
    classifier = stance_classifier.StanceClassifier()
    print("--- %s seconds ---" % (time.time() - start_time))
    print(classifier.classify(reply)) # result

print("--- %s seconds ---" % (time.time() - start_time))
