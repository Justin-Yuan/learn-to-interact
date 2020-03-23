import numpy as np 
import torch 




class RolloutState(object):
    """ container for multi-agent controller state during rollout 
    could store information such as grouping, beliefs, etc.
    """
    def __init__(self, scheme, batch_size, **kwargs):
        self.scheme = scheme 
        self.batch_size = batch_size

