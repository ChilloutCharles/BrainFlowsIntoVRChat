from collections import ChainMap

from logic.power_ratios import Power_Ratios
from constants import BAND_POWERS
from utils import tanh_normalize, map2dto1d

class Focus_Relax(Power_Ratios):
    def __init__(self, board, window_seconds=2, normalize_scale=1.1, filter_period=1):
        super().__init__(board, window_seconds=window_seconds, filter_period=filter_period)
        self.bands = [BAND_POWERS.Alpha, BAND_POWERS.Beta, BAND_POWERS.Theta]
        self.band_names = list(map(lambda b: b.name.lower(), self.bands))
        self.locations = ["left", "right", "avg"]
        self.normalize_scale = normalize_scale

    def get_data_dict(self):
        ret_dict = super().get_data_dict()
        
        # filter dict down to alpha, beta, and theta powers
        def endsWithBandnames(osc_name):
            return any(map(lambda b: osc_name.endswith(b), self.band_names))
        fr_dict = {k:ret_dict[k] for k in ret_dict if endsWithBandnames(k)}

        # calculate focus and relax per location
        def calculate_location(location):
            # keys will be sorted in alphabetical order: alpha, beta, theta
            loc_keys = sorted(filter(lambda k: location in k, fr_dict.keys()))
            
            loc_powers = [fr_dict[k] for k in loc_keys]
            focus = tanh_normalize(loc_powers[1] / loc_powers[2], self.normalize_scale, -1)
            relax = tanh_normalize(loc_powers[0] / loc_powers[2], self.normalize_scale, -1)

            loc_dict = {
                "_".join(("osc", "focus", location)) : focus,
                "_".join(("osc", "relax", location)) : relax
            }

            # get hueshift parameter from the avg focus and relax
            if location is "avg":
                loc_dict["HueShift"] = map2dto1d(focus + 1, relax + 1, 2) / 8

            return loc_dict
        loc_dicts = list(map(calculate_location, self.locations)) + [ret_dict]

        return dict(ChainMap(*loc_dicts))