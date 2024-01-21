from abc import ABC

class Base_Logic(ABC):
    def __init__(self, board):
        self.board = board

    def get_data_dict(self):
        ...
    