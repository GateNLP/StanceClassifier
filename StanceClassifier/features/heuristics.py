import re

class Heuristics:

    def pattern1(reply):
        pattern = ".*is (that|this|it) true.*"
        result = re.match(pattern, reply)
        if result != None:
            return 1.
        return 0.
    
    def pattern2(reply):
        pattern = ".*wh[a]*t[\\?!].*"
        result = re.match(pattern, reply)
        if result != None:
            return 1.
        return 0.
    
    def pattern3(reply):
        pattern = ".*(real\\?|really\\?).*"
        result = re.match(pattern, reply)
        if result != None:
            return 1.
        return 0.
    
    def pattern4(reply):
        pattern = ".*fact check.*"
        result = re.match(pattern, reply)
        if result != None:
            return 1.
        return 0.
    
    def pattern5(reply):
        pattern = ".*scandal.*"
        result = re.match(pattern, reply)
        if result != None:
            return 1.
        return 0.
    
    def pattern6(reply):
        pattern = ".*(rumor?|debunk?).*"
        result = re.match(pattern, reply)
        if result != None:
            return 1.
        return 0.
    
    def pattern7(reply):
        pattern = ".*(that|this|it) is not true.*"
        result = re.match(pattern, reply)
        if result != None:
            return 1.
        return 0.

    def pattern8(reply):
        pattern = ".*(not confirmed|unconfirmed).*"
        result = re.match(pattern, reply)
        if result != None:
            return 1.
        return 0.

    def pattern9(reply):
        pattern = ".*(hopeful??|still).*"
        result = re.match(pattern, reply)
        if result != None:
            return 1.
        return 0.

    def pattern10(reply):
        pattern = ".*(stop) .*"
        result = re.match(pattern, reply)
        if result != None:
            return 1.
        return 0.
