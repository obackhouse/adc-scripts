'''
ADC(2) for electron affinities for restricted references with density fitting.
'''

import numpy as np
from adc import ip_df_radc2
from pyscf import lib


class ADCHelper(ip_df_radc2.ADCHelper):
    def build(self):
        self.swap_ov()
        ip_df_radc2.ADCHelper.build(self)
