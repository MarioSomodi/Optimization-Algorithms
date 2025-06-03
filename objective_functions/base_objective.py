from abc import ABC, abstractmethod

class BaseObjective(ABC):

    def __init__(self, name: str, bounds: tuple, global_min=None):
        self.name = name
        self.bounds = bounds
        self.global_min = global_min

    @abstractmethod
    def evaluate(self, x):
        pass