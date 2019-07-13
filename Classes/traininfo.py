# Train Info is a class meant to follow filter stats and how accurate 
# the Neural Network is. 

class TrainInfo:
    def __init__(self, name):
        self.name = name
        self.filters = []
