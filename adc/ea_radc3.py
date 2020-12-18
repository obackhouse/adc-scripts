'''
ADC(3) for electron affinities for restricted references.
'''

import numpy as np
from adc import ip_radc3
from pyscf import lib


class ADCHelper(ip_radc3.ADCHelper):
    def build(self):
        self.swap_ov()
        ip_radc3.ADCHelper.build(self)
        self.sign = -1
