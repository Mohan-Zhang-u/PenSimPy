from pensimpy.pensim_classes.Channel import Channel


class Xinterp:
    """
    Noise added by randn and filtered be a low pass filter
    """

    def __init__(self):
        self.distMuP = ''
        self.distMuX = ''
        self.distcs = ''
        self.distcoil = ''
        self.distabc = ''
        self.distPAA = ''
        self.distTcin = ''
        self.distO_2in = ''
        # extra
        self.NH3_shots = Channel()
        # hard code
        self.NH3_shots.y = [0] * 2000
