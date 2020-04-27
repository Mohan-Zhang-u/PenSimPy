import numpy as np


class BatchRunFlags:
    """
    Params for each batch; Batches are independent.
    """

    def __init__(self, Num_of_Batches):
        self.Num_of_Batches = Num_of_Batches
        self.Batch_fault_order_reference = np.zeros((self.Num_of_Batches, 1), dtype=float)
        self.Control_strategy = np.ones((1, self.Num_of_Batches), dtype=int)
        self.Batch_length = np.ones((1, self.Num_of_Batches), dtype=int)
        self.Raman_spec = np.zeros((1, self.Num_of_Batches), dtype=int)
