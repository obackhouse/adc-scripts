'''
ADC(2) for ionization potentials for restricted references with density fitting.
'''

import numpy as np
from adc import utils, mpi_helper, ip_radc2
from pyscf import lib


def get_1h(helper):
    Lov, Loo, eia, eija = helper.unpack()
    nocc, nvir = helper.nocc, helper.nvir
    eija = eija.reshape(nocc, -1)

    h1 = np.zeros((nocc, nocc))
    ejab = lib.direct_sum('ja,b->jab', eia, -helper.ev)
    for i in mpi_helper.distr_iter(range(nocc)):
        eiajb = (helper.eo[i] + ejab).reshape(nocc, -1)
        iba = np.dot(Lov.reshape(-1, nocc*nvir).T, Lov[:,i]).reshape(nocc, -1)
        iab = iba.reshape(nocc, nvir, nvir).swapaxes(1,2).reshape(nocc, -1)
        t2 = iab / eiajb
        v = 2.0 * iab - iba
        h1 += np.dot(v, t2.T) * 0.5

    mpi_helper.barrier()
    mpi_helper.allreduce_inplace(h1)

    h1 += h1.T
    h1 += np.diag(helper.eo)

    return h1


def get_matvec(helper):
    Lov, Loo, eia, eija = helper.unpack()
    nocc, nvir = helper.nocc, helper.nvir
    eija = eija.reshape(nocc, -1)

    h1 = get_1h(helper)

    def matvec(y):
        y = np.asarray(y, order='C')
        r = np.zeros_like(y)

        yi = y[:nocc]
        ri = r[:nocc]
        yija = y[nocc:].reshape(nocc, -1)
        rija = r[nocc:].reshape(nocc, -1)

        for i in mpi_helper.distr_iter(range(nocc)):
            kja = np.dot(Loo[:,i].T, Lov.reshape(-1, nocc*nvir))
            kia = np.dot(Loo.reshape(-1, nocc*nocc).T, Lov[:,i])
            kia = kia.reshape(nocc, -1)
            v = 2.0 * kja - kia
            ri += np.dot(kja, yija[i])
            rija[i] += np.dot(v.T, yi)
            rija[i] += eija[i] * yija[i]

        mpi_helper.barrier()
        mpi_helper.allreduce_inplace(ri)
        mpi_helper.allreduce_inplace(rija)

        ri += np.dot(h1, yi)

        return r

    diag = np.concatenate([np.diag(h1), eija.ravel()])

    return matvec, diag

class ADCHelper(ip_radc2.ADCHelper):
    def build(self):
        self.eo, self.ev = self.e[self.o], self.e[self.v]
        self.co, self.cv = self.c[:,self.o], self.c[:,self.v]

        self.Lov = self.ao2mo(self.co, self.cv)
        self.Loo = self.ao2mo(self.co, self.co)

        self.eia = lib.direct_sum('i,a->ia', self.eo, -self.ev)
        self.eija = lib.direct_sum('i,ja->ija', self.eo, self.eia)

        self._to_unpack = ['Lov', 'Loo', 'eia', 'eija']

    def mp2(self):
        eia, Lov = self.eia, self.Lov
        nocc, nvir = self.nocc, self.nvir

        ejab = lib.direct_sum('ja,b->jab', eia, -self.ev)
        e_mp2 = 0.0
        for i in mpi_helper.distr_iter(range(nocc)):
            eiajb = self.eo[i] + ejab
            iba = np.dot(Lov.reshape(-1, nocc*nvir).T, Lov[:,i])
            iba = iba.reshape(nocc, nvir, nvir)
            iab = iba.swapaxes(1,2)
            t2 = iab / eiajb
            e_mp2 += np.sum(iab * t2) * 2.0
            e_mp2 -= np.sum(iba * t2)

        mpi_helper.barrier()
        e_mp2 = mpi_helper.allreduce(e_mp2)

        return e_mp2

    get_matvec = get_matvec
    get_1h = get_1h
