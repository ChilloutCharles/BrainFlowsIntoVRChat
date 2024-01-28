from logic.base_logic import Base_Logic
from logic.neuro_feedback import Neuro_Feedback
from utils import map2dto1d

class Addons(Base_Logic):
    def __init__(self, board, logic_name="addons", window_seconds=2, normalize_scale=1.1, ema_decay=0.025):
        super().__init__(board, logic_name)
        self.neuro_feedback_logic = Neuro_Feedback(board, logic_name, window_seconds=window_seconds,
            normalize_scale=normalize_scale, ema_decay=ema_decay)
    
    def get_data_dict(self):
        # get neurofeedback scores 
        nf_dict_all = self.neuro_feedback_logic.get_data_dict()

        # get average scores
        nf_dict_avg = nf_dict_all['avg']
        focus = nf_dict_avg['focus']
        relax = nf_dict_avg['relax']

        # remap focus and relax to 1D
        hueshift = map2dto1d(focus + 1, relax + 1, 2) / 8

        return {
            "HueShift": hueshift
        }