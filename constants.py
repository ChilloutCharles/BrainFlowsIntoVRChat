import enum

VERSION_MAJOR = 1
VERSION_MINOR = 0

OSC_BASE_PATH = '/avatar/parameters/'
BFI_ROOT = 'BFI/'

class BAND_POWERS(enum.IntEnum):
    Gamma = 4
    Beta = 3
    Alpha = 2
    Theta = 1
    Delta = 0