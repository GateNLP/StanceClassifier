import socket, json
import sys
from struct import pack
sys.path[0:0] = ["util/"]
from util import Util

#Load configurations
util = Util()
configurations = util.loadResources('configurations.txt')



source = json.load(open("examples/source"))
reply = json.load(open("examples/reply"))

info = {}
info['source'] = source
info['reply'] = reply



data = json.dumps(info)

print(len(data.encode('utf-8')))

s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.connect((configurations['local_server_hostname'],int(configurations['local_server_port'])))

print("Sending data in batches...")
length = pack('>Q', len(data.encode('utf-8')))
s.sendall(length)
s.sendall(data.encode('utf-8'))
ack = s.recv(1)
if ack == b'\x00':
    print("Data sent successfully.")
response = json.loads(s.recv(1024).decode('utf-8'))
s.close()
print("*** Results ***")
print("Class: " + str(response['class']))
print("Probabilities: " + str(response['probs']))
