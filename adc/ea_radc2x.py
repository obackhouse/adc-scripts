'''
ADC(2)-x for electron affinities for restricted references.
'''

import numpy as np
from adc import ip_radc2x
from pyscf import lib


class ADCHelper(ip_radc2x.ADCHelper):
    def build(self):
        self.swap_ov()
        ip_radc2x.ADCHelper.build(self)
        self.sign = -1
