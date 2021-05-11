'''
ADC(2) for ionization potentials for restricted references.
'''

import numpy as np
from adc import utils, mpi_helper
from pyscf import lib


def get_1h(helper):
    t2, ovov, ooov, eija = helper.unpack()
    nocc, nvir = helper.nocc, helper.nvir

    p0, p1 = mpi_helper.distr_blocks(nocc*nvir**2)
    t2_block = t2.reshape(nocc, -1)[:,p0:p1]
    ovov_as_block  = ovov.reshape(nocc, -1)[:,p0:p1] * 2.0
    ovov_as_block -= ovov.swapaxes(1,3).reshape(nocc, -1)[:,p0:p1]

    h1  = np.dot(t2_block, ovov_as_block.T) * 0.5
    h1 += h1.T

    mpi_helper.barrier()
    mpi_helper.allreduce_inplace(h1)

    h1 += np.diag(helper.eo)

    return h1


def get_matvec(helper):
    t2, ovov, ooov, eija = helper.unpack()
    nocc, nvir = helper.nocc, helper.nvir

    p0, p1 = mpi_helper.distr_blocks(nocc*nvir**2)
    t2_block = t2.reshape(nocc, -1)[:,p0:p1]
    ovov_as_block  = ovov.reshape(nocc, -1)[:,p0:p1] * 2.0
    ovov_as_block -= ovov.swapaxes(1,3).reshape(nocc, -1)[:,p0:p1]

    q0, q1 = mpi_helper.distr_blocks(nocc**2*nvir)
    ooov_block = ooov.reshape(nocc, -1)[:,q0:q1]
    ooov_as_block  = ooov_block * 2.0
    ooov_as_block -= ooov.swapaxes(1,2).reshape(nocc, -1)[:,q0:q1]
    eija_block = eija.ravel()[q0:q1]

    h1 = get_1h(helper)

    def matvec(y):
        y = np.asarray(y, order='C')
        r = np.zeros_like(y)

        yi = y[:nocc]
        ri = r[:nocc]
        yija = y[nocc:]
        rija = r[nocc:]

        yija_block = yija[q0:q1]
        rija_block = rija[q0:q1]

        ri += np.dot(ooov_block, yija_block)
        rija_block += np.dot(yi, ooov_as_block)
        rija_block += eija_block * yija_block

        mpi_helper.barrier()
        mpi_helper.allreduce_inplace(ri)
        mpi_helper.allreduce_inplace(rija)

        ri += np.dot(h1, yi)

        return r

    diag = np.concatenate([np.diag(h1), eija.ravel()])

    return matvec, diag

def get_guesses(helper, diag, nroots, koopmans=False):
    guesses = np.zeros((nroots, diag.size))

    if koopmans:
        arg = np.argsort(np.absolute(diag[:helper.nocc]))
        nroots = min(nroots, helper.nocc)
    else:
        arg = np.argsort(np.absolute(diag))

    for root, guess in enumerate(arg[:nroots]):
        guesses[root,guess] = 1.0

    return list(guesses)

def get_moments(helper, nmax):
    t2, ovov, ooov, eija = helper.unpack()
    nocc, nvir = helper.nocc, helper.nvir

    vl = 2.0 * ooov - ooov.swapaxes(1,2)
    vr = ooov

    vl = vl.reshape(nocc, -1)
    vr = vr.reshape(nocc, -1)

    t = np.zeros((nmax+1, nocc, nocc), dtype=ovov.dtype)

    for n in range(nmax+1):
        t[n] = np.dot(vl, vr.T.conj())
        if n != nmax:
            vl *= eija.ravel()[None]

    return t

class ADCHelper(utils._ADCHelper):
    def build(self):
        self.eo, self.ev = self.e[self.o], self.e[self.v]
        self.co, self.cv = self.c[:,self.o], self.c[:,self.v]

        self.ovov = self.ao2mo(self.co, self.cv, self.co, self.cv)
        self.ooov = self.ao2mo(self.co, self.co, self.co, self.cv)

        self.eija = lib.direct_sum('i,j,a->ija', self.eo, self.eo, -self.ev)

        eiajb = lib.direct_sum('i,a,j,b->iajb', self.eo, -self.ev, self.eo, -self.ev)
        self.t2 = self.ovov / eiajb

        self._to_unpack = ['t2', 'ovov', 'ooov', 'eija']

    get_matvec = get_matvec
    get_guesses = get_guesses
    get_1h = get_1h
    get_moments = get_moments
