'''
ADC(3) for electron affinities for restricted references.
'''

from adc import ip_radc3


class ADCHelper(ip_radc3.ADCHelper):
    def build(self):
        self.swap_ov()
        ip_radc3.ADCHelper.build(self)
        self.sign = -1
