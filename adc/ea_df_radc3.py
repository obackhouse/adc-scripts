'''
ADC(3) for electron affinities for restricted references with density fitting.
'''

from adc import ip_df_radc3


class ADCHelper(ip_df_radc3.ADCHelper):
    def build(self):
        self.swap_ov()
        ip_df_radc3.ADCHelper.build(self)
        self.sign = -1
