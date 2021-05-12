'''
ADC(2)-x for electron affinities for restricted references with density fitting.
'''

from adc import ip_df_radc2x


class ADCHelper(ip_df_radc2x.ADCHelper):
    def build(self):
        self.swap_ov()
        ip_df_radc2x.ADCHelper.build(self)
        self.sign = -1
