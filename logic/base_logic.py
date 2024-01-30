from abc import ABC

class BaseLogic(ABC):
    def __init__(self, board):
        self.board = board

    def get_data_dict(self):
        return {}

class OptionalBaseLogic(BaseLogic):
    SUPPORTED = "Supported"

    def __init__(self, board, supported=False):
        super().__init__(board)
        self.supported = supported
    
    def get_data_dict(self):
        return {OptionalBaseLogic.SUPPORTED: self.supported}
    