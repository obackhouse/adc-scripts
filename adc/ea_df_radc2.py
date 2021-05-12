'''
ADC(2) for electron affinities for restricted references with density fitting.
'''

from adc import ip_df_radc2


class ADCHelper(ip_df_radc2.ADCHelper):
    def build(self):
        self.swap_ov()
        ip_df_radc2.ADCHelper.build(self)
