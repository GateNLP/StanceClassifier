import urllib.request
import json
from StanceClassifier.util import *


info = {}
info['original'] = json.load(open(path_from_root("examples/source")))
info['reply'] = json.load(open(path_from_root("examples/reply")))

myurl = "http://localhost:7272/"
req = urllib.request.Request(myurl)
req.add_header('Content-Type', 'application/json; charset=utf-8')
jsondata = json.dumps(info)
jsondataasbytes = jsondata.encode('utf-8')   # needs to be bytes
req.add_header('Content-Length', len(jsondataasbytes))
content = urllib.request.urlopen(req, jsondataasbytes).read()



print(content)
