from abc import ABC

class Base_Logic(ABC):
    def __init__(self, board, logic_name):
        self.board = board
        self.logic_name = logic_name

    def get_data_dict(self):
        return {}
    
    def get_logic_name(self):
        return self.logic_name
    