from abc import ABC

class BaseLogic(ABC):
    def __init__(self, board):
        self.board = board

    def get_data_dict(self):
        return {}
    