'''
ADC(2)-x for electron affinities for restricted references.
'''

from adc import ip_radc2x


class ADCHelper(ip_radc2x.ADCHelper):
    def build(self):
        self.swap_ov()
        ip_radc2x.ADCHelper.build(self)
        self.sign = -1
