'''
ADC(2) for ionization potentials for restriced periodic (k-space) references.
'''

import numpy as np
import itertools
from adc2 import utils, mpi_helper, ip_radc2
from pyscf import lib
from pyscf.pbc import tools
from pyscf.pbc.mp.kmp2 import _padding_k_idx

#TODO: handle padded calculations


def get_matvec(helper, ki):
    t2, ovov, ooov, eija = helper.t2, helper.ovov, helper.ooov, helper.eija
    nocc, nvir = max(helper.nocc), max(helper.nvir)
    nkpts = helper.nkpts

    h1 = np.zeros((nocc, nocc), dtype=helper.dtype)

    for kj, kk in mpi_helper.distr_iter(helper.kpt_loop(2)):
        kl = helper.kconserv[ki,kj,kk]
        vk = 2.0 * ovov[ki,kj,kk] - ovov[ki,kl,kk].swapaxes(1,3)
        vk = vk.reshape(nocc, -1)
        t2k = t2[ki,kj,kk].reshape(nocc, -1)
        h1 += np.dot(t2k, vk.T.conj()) * 0.5

    mpi_helper.barrier()
    mpi_helper.allreduce_inplace(h1)

    h1 += h1.T.conj()
    h1 += np.diag(helper.eo[ki])

    def matvec(y):
        y = np.asarray(y, order='C', dtype=helper.dtype)
        r = np.zeros_like(y)

        yi = y[:nocc]
        ri = r[:nocc]
        yija = y[nocc:].reshape(nkpts, nkpts, -1)
        rija = r[nocc:].reshape(nkpts, nkpts, -1)

        for kj, kk in helper.kpt_loop(nk=2):
            kl = helper.kconserv[ki,kj,kk]
            vk = 2.0 * ooov[ki,kj,kk] - ooov[ki,kk,kj].swapaxes(1,2) # correct?
            vk = vk.reshape(nocc, -1)
            ri += np.dot(ooov[ki,kj,kk].reshape(nocc, -1), yija[kj,kk])
            rija[kj,kk] += np.dot(yi, vk.conj())
            rija[kj,kk] += eija[ki,kj,kk].ravel() * yija[kj,kk]

        mpi_helper.barrier()
        mpi_helper.allreduce_inplace(ri)
        mpi_helper.allreduce_inplace(rija)

        ri += np.dot(h1, yi)

        return r

    diag = np.concatenate([np.diag(h1), eija[ki].ravel()])

    return matvec, diag

class ADCHelper(ip_radc2.ADCHelper):
    def build(self):
        nmo_per_kpt = [x.size for x in self.o]
        nocc_per_kpt = [np.sum(x) for x in self.o]
        nmo, nocc = max(nmo_per_kpt), max(nocc_per_kpt)
        nvir = nmo - nocc

        self.opad, self.vpad = _padding_k_idx(nmo_per_kpt, nocc_per_kpt, kind='split')
        self.kconserv = tools.get_kconserv(self.mf.cell, self.mf.kpts)
        dtype = np.complex128 #TODO

        self.eo = [e[o] for e,o in zip(self.e, self.o)]
        self.ev = [e[v] for e,v in zip(self.e, self.v)]
        self.co = [c[:,o] for c,o in zip(self.c, self.o)]
        self.cv = [c[:,v] for c,v in zip(self.c, self.v)]

        self.eo = np.array([e[x] for e,x in zip(self.eo, self.opad)])
        self.ev = np.array([e[x] for e,x in zip(self.ev, self.vpad)])
        self.co = np.array([c[:,x] for c,x in zip(self.co, self.opad)])
        self.cv = np.array([c[:,x] for c,x in zip(self.cv, self.vpad)])

        self.ovov = np.zeros((self.nkpts,)*3 + (nocc, nvir, nocc, nvir), dtype=dtype)
        self.ooov = np.zeros((self.nkpts,)*3 + (nocc, nocc, nocc, nvir), dtype=dtype)
        self.eija = np.zeros((self.nkpts,)*3 + (nocc, nocc, nvir))
        self.t2 = self.ovov.copy()

        for ki, kj, kk in mpi_helper.distr_iter(self.kpt_loop(3)):
            kl = self.kconserv[ki,kj,kk]
            kpts = [ki, kj, kk, kl]
            eo, ev, co, cv = self.eo, self.ev, self.co, self.cv

            self.ovov[ki,kj,kk] = self.ao2mo(kpts, co[ki], cv[kj], co[kk], cv[kl])
            self.ooov[ki,kj,kk] = self.ao2mo(kpts, co[ki], co[kj], co[kk], cv[kl])
            self.eija[ki,kj,kk] = lib.direct_sum('i,j,a->ija', eo[kj], eo[kk], -ev[kl])
            eiajb = lib.direct_sum('i,a,j,b->iajb', eo[ki], -ev[kj], eo[kk], -ev[kl])
            self.t2[ki,kj,kk] = self.ovov[ki,kj,kk] / eiajb

        mpi_helper.barrier()
        mpi_helper.allreduce_inplace(self.ovov)
        mpi_helper.allreduce_inplace(self.ooov)
        mpi_helper.allreduce_inplace(self.eija)
        mpi_helper.allreduce_inplace(self.t2)

    def ao2mo(self, kpts, *coeffs):
        eri = self.mf.with_df.ao2mo(coeffs, kpts=self.mf.kpts[kpts], compact=False)
        eri = eri.reshape([x.shape[1] for x in coeffs]) / self.nkpts
        return eri

    def kpt_loop(self, nk):
        return itertools.product(range(self.nkpts), repeat=nk)

    def mp2(self):
        e_mp2 = 0.0

        for ki, kj, kk in mpi_helper.distr_iter(self.kpt_loop(3)):
            t2 = self.t2[ki,kj,kk]
            e_mp2 += np.sum(self.t2 * self.ovov[ki,kj,kk].conj()) * 2.0
            e_mp2 -= np.sum(self.t2 * self.ovov[kk,kj,ki].conj().swapaxes(0,2))

        e_mp2 = e_mp2.real

        mpi_helper.barrier()
        e_mp2 = mpi_helper.allreduce(e_mp2)

        return e_mp2 / self.nkpts

    @property
    def nkpts(self):
        return len(self.e)
    @property
    def dtype(self):
        return self.t2.dtype

    get_matvec = get_matvec
