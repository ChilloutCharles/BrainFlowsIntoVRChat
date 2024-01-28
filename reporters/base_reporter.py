from abc import ABC

class Base_Reporter(ABC):
    def send(self, data_dict):
        ...