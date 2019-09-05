from .extract_features import Features
import json
import sys

def open_json(f):
    with open(f) as json_file:
        data = json.load(json_file)
    return data
    
def loadResources(path):
    #Open resource file:
    f = open(path)
     
    #Create resource map:
    resources = {}
     
    #Read resource paths:
    for line in f:
        data = line.strip().split('|||')
        if data[0] in resources:
            print('Repeated resource name: ' + data[0] + '. Please change the name of this resource.')
        resources[data[0]] = data[1]
    f.close()
    #Return resource database:
    return resources

#Load resources:
print("Loading resources")
resources = loadResources('../resources.txt')
#configurations = loadResources('../configurations.txt')
print("Done. %d resources added!" % len(resources.keys()))

f = sys.argv[1]
f2 = sys.argv[2]
f_json = open_json(f)
f_json2 = open_json(f2)

feat = Features(resources)

feat_set = feat.features(f_json, f_json2)
print(feat_set)


