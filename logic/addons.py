from logic.base_logic import BaseLogic
from logic.neuro_feedback import NeuroFB
from utils import map2dto1d

class Addons(BaseLogic):
    def __init__(self, board, window_seconds=2, normalize_scale=1.1, ema_decay=0.025):
        super().__init__(board)
        self.neuro_feedback_logic = NeuroFB(board, window_seconds=window_seconds,
            normalize_scale=normalize_scale, ema_decay=ema_decay)
    
    def get_data_dict(self):
        # get neurofeedback scores 
        nf_dict = self.neuro_feedback_logic.get_data_dict()

        # get average scores
        focus = nf_dict[NeuroFB.FOCUS + NeuroFB.AVERAGE + NeuroFB.SIGNED]
        relax = nf_dict[NeuroFB.RELAX + NeuroFB.AVERAGE + NeuroFB.SIGNED]

        # remap focus and relax to 1D
        hueshift = map2dto1d(focus + 1, relax + 1, 2) / 8

        return {
            "HueShift": hueshift
        }