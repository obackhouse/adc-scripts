'''
ADC(2) for electronic affinities for restricted periodic (k-space) references.
'''

import numpy as np
from adc import ip_kradc2
from pyscf import lib


class ADCHelper(ip_kradc2.ADCHelper):
    def build(self):
        self.swap_ov()
        ip_kradc2.ADCHelper.build(self)
