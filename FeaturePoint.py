# simple class structure to easily query over a list of feature points locations and their descriptors

class FeaturePoint:
     def __init__(self, position, descriptor):
        self.position = position
        self.descriptor = descriptor
        
