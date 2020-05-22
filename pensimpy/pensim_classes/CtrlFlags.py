class CtrlFlags:
    """
    Control flags.
    """

    def __init__(self, **kwargs):
        self.SBC = 0
        self.PRBS = kwargs['PRBS']
        self.Fixed_Batch_length = kwargs['Fixed_Batch_length']
        self.IC = 0
        self.Inhib = 2
        self.Dis = 1
        self.Faults = kwargs['Faults']
        self.Vis = 0
        self.Raman_spec = kwargs['Raman_spec']
        self.Batch_Num = kwargs['Batch_Num']
        self.Off_line_m = 12
        self.Off_line_delay = 4
        self.plots = 1
        self.T_sp = 0
        self.pH_sp = 0
