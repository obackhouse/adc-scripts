'''
ADC(2) for electronic affinities for restricted periodic (k-space) references.
'''

from adc import ip_kradc2


class ADCHelper(ip_kradc2.ADCHelper):
    def build(self):
        self.swap_ov()
        ip_kradc2.ADCHelper.build(self)
