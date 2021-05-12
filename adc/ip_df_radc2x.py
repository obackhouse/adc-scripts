'''
ADC(2)-x for ionization potentials for restricted references.
'''

import numpy as np
from adc import utils, mpi_helper, ip_df_radc2
from pyscf import lib


def get_1h(helper):
    Lov, Loo, Lvv, eia, eija = helper.unpack()
    nocc, nvir = helper.nocc, helper.nvir

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
    Lov, Loo, Lvv, eia, eija = helper.unpack()
    nocc, nvir = helper.nocc, helper.nvir
    sign = helper.sign

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
            ri += np.dot(v, yija[i])
            rija[i] += np.dot(kja.T, yi)
            rija[i] += eija[i].ravel() * yija[i]

        yija = yija.reshape(nocc, nocc, nvir)
        rija = rija.reshape(nocc, nocc, nvir)
        yija_as = 2.0 * yija - yija.swapaxes(0,1)

        for l in mpi_helper.distr_iter(range(nocc)):
            o_vv = np.dot(Loo[:,:,l].T, Lvv.reshape(-1, nvir*nvir)).reshape(nocc, nvir, nvir)
            ov_v = np.dot(Lov.reshape(-1, nocc*nvir).T, Lov[:,l]).reshape(nocc, nvir, nvir)
            ooo_ = np.dot(Loo.reshape(-1, nocc*nocc).T, Loo[:,:,l]).reshape(nocc, nocc, nocc)

            rija -= utils.einsum('ikj,ka->ija', ooo_, yija[:,l]) * sign
            rija += utils.einsum('iba,jb->ija', o_vv, yija[l]) * sign
            rija -= utils.einsum('jab,ib->ija', ov_v, yija_as[:,l]) * sign
            rija += utils.einsum('jba,ib->ija', o_vv, yija[:,l]) * sign

        mpi_helper.barrier()
        mpi_helper.allreduce_inplace(ri)
        mpi_helper.allreduce_inplace(rija)

        ri += np.dot(h1, yi)

        return r

    diag = np.concatenate([np.diag(h1), eija.ravel()])

    if helper.guess_high_order:
        # According to A. Sokolov these might actually make things worse
        # See https://github.com/pyscf/pyscf/commit/994e325159866bc74319418033db270a6b6a9d57#r45037621
        diag_ija = diag[nocc:].reshape(nocc, nocc, nvir)
        diag_ija -= utils.einsum('Lii,Ljj->ij', Loo, Loo)[:,:,None] * sign
        diag_ija += utils.einsum('Ljj,Laa->ja', Loo, Lvv)[None,:,:] * sign
        diag_ija += utils.einsum('Lii,Laa->ia', Loo, Lvv)[:,None,:] * sign

    return matvec, diag


class ADCHelper(ip_df_radc2.ADCHelper):
    def build(self):
        self.eo, self.ev = self.e[self.o], self.e[self.v]
        self.co, self.cv = self.c[:,self.o], self.c[:,self.v]

        self.Lov = self.ao2mo(self.co, self.cv)
        self.Loo = self.ao2mo(self.co, self.co)
        self.Lvv = self.ao2mo(self.cv, self.cv)

        self.eia = lib.direct_sum('i,a->ia', self.eo, -self.ev)
        self.eija = lib.direct_sum('i,ja->ija', self.eo, self.eia)

        self.sign = 1
        self.guess_high_order = True

        self._to_unpack = ['Lov', 'Loo', 'Lvv', 'eia', 'eija']

    get_matvec = get_matvec
    get_1h = get_1h
