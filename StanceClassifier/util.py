import os
from pathlib import Path


def get_project_root():
    return Path(__file__).parent.parent

def path_from_root(path):
    """
    Appends the path to the project's root path obtained from get_project_root()
    :param path: Relative path that will be joined to the root path of the project
    :return:
    """
    return os.path.join(get_project_root(), path)

class Util():

    #def __init__(self):


    def loadResources(self, path):
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

