'''
ADC(2) for electron affinities for restricted references.
'''

from adc import ip_radc2


class ADCHelper(ip_radc2.ADCHelper):
    def build(self):
        self.swap_ov()
        ip_radc2.ADCHelper.build(self)
