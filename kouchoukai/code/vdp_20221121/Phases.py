from scipy import signal
import numpy as np

class Phase_transform:
    def __init__(self):
        self.M = 50

    def calculate_Sm(self, proto_phase):
        """
        ptoro_phase分布のフーリエ係数を近似的に計算, 
        詳しくは Kralemann, 2007, 2008, 2014
        """
        Sm_array = np.ones(self.M, dtype=np.complex128)
        for m in range(1, self.M + 1):
            Sm_array[int(m-1)] = np.mean(np.exp(-1j * m * proto_phase))
        return Sm_array

    def kralemann_phase(self, proto_phase, Sm_array):
        """
        kralemann_method: calculate_Smで計算したフーリエ係数Smを用いて, protophaseを位相に変換
            input:
                (ndarray) proto_phase: 1d-ndarray
            return:
                (ndarray) phase: 1d-ndarray
        """
        phase = np.copy(proto_phase)
        for m in range(1, self.M + 1):
            Sm = Sm_array[int(m-1)]
            phase += 2.0 * np.imag((Sm / m) * (np.exp(1j * m * proto_phase) - 1.0))
        return phase