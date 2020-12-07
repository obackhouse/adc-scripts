'''
ADC(2) for electron affinities for restricted references.
'''

import numpy as np
from adc2 import ip_radc2
from pyscf import lib


class ADCHelper(ip_radc2.ADCHelper):
    def build(self):
        self.swap_ov()
        ip_radc2.ADCHelper.build(self)
