'''
Utility functions.
'''

import numpy as np
import functools
from pyscf import lib, ao2mo
from adc2 import mpi_helper

einsum = lib.einsum
#einsum = functools.partial(np.einsum, optimize=True)

def nested_apply(item, func, ndim=1):
    peek_in = item
    for i in range(ndim):
        peek_in = peek_in[0]
    if isinstance(peek_in, (tuple, list, np.ndarray)):
        return tuple(nested_apply(x, func) for x in item)
    else:
        return func(item)

class _ADCHelper:
    def __init__(self, mf):
        if not mf.converged:
            raise ValueError('Mean-field method is not converged')
        self.mf = mf
        self._build_common()
        self.build()

    def _build_common(self):
        self.e = self.mf.mo_energy
        self.c = self.mf.mo_coeff
        self.o = nested_apply(self.mf.mo_occ, lambda x: x > 0)
        self.v = nested_apply(self.mf.mo_occ, lambda x: x == 0)

    def swap_ov(self):
        self.o, self.v = self.v, self.o
        #self.eo, self.ev = self.ev, self.eo
        #self.co, self.cv = self.cv, self.co

    def ao2mo(self, *coeffs):
        if not hasattr(self.mf, 'with_df'):
            eri = ao2mo.incore.general(self.mf._eri, coeffs, compact=False)
            eri = eri.reshape([x.shape[1] for x in coeffs])
        else:
            eri = lib.unpack_tril(self.mf.with_df._cderi, axis=-1) #TODO improve
            eri = einsum('Lpq,pi,qj->Lij', eri, *coeffs)
        return eri

    def mp2(self):
        e_mp2 = 0.0

        for i in mpi_helper.distr_iter(range(self.nocc)):
            t2 = self.t2[i]
            ovov = self.ovov[i]
            e_mp2 += np.sum(t2 * ovov) * 2.0
            e_mp2 -= np.sum(t2 * ovov.swapaxes(0,2))

        mpi_helper.barrier()
        e_mp2 = mpi_helper.allreduce(e_mp2)

        return e_mp2

    def build(self):
        raise AttributeError

    def get_matvec(self):
        raise AttributeError

    def get_guesses(self):
        raise AttributeError

    @property
    def nocc(self):
        return nested_apply(self.eo, len)
    @property
    def nvir(self):
        return nested_apply(self.ev, len)
    @property
    def nmo(self):
        return nested_apply(self.e, len)

