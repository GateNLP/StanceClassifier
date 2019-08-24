
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

