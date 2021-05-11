'''
ADC(2)-x for ionization potentials for restricted references.
'''

import numpy as np
from adc import utils, mpi_helper, ip_radc2
from pyscf import lib


def get_1h(helper):
    t2, ovov, ooov, oooo, oovv, eija = helper.unpack()
    nocc, nvir = helper.nocc, helper.nvir
    sign = helper.sign

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
    t2, ovov, ooov, oooo, oovv, eija = helper.unpack()
    nocc, nvir = helper.nocc, helper.nvir
    sign = helper.sign

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

        ri += np.dot(ooov_as_block, yija_block)
        rija_block += np.dot(yi, ooov_block)
        rija_block += eija_block * yija_block

        #TODO: really not a fan of this... mixing and matching MPI strategies
        p0, p1 = mpi_helper.distr_blocks(nocc)
        yija = yija.reshape(nocc, nocc, nvir)
        yija_as = 2.0 * yija - yija.swapaxes(0,1)
        rija_block = rija.reshape(nocc, nocc, nvir)[p0:p1]

        rija_block -= utils.einsum('ikjl,kla->ija', oooo[p0:p1], yija) * sign
        rija_block += utils.einsum('ilba,ljb->ija', oovv[p0:p1], yija) * sign
        rija_block -= utils.einsum('jalb,ilb->ija', ovov, yija_as[p0:p1]) * sign
        rija_block += utils.einsum('jlba,ilb->ija', oovv, yija[p0:p1]) * sign

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
        diag_ija -= utils.einsum('iijj->ij', oooo)[:,:,None] * sign
        diag_ija += utils.einsum('jjaa->ja', oovv)[None,:,:] * sign
        diag_ija += utils.einsum('iiaa->ia', oovv)[:,None,:] * sign

    return matvec, diag


def get_moments(helper, nmax):
    t2, ovov, ooov, oooo, oovv, eija = helper.unpack()
    nocc, nvir = helper.nocc, helper.nvir
    sign = helper.sign

    vl = 2.0 * ooov - ooov.swapaxes(1,2)
    vr = ooov.copy()

    t = np.zeros((nmax+1, nocc, nocc), dtype=ovov.dtype)
    t[0] = np.dot(vl.reshape(nocc, -1), vr.reshape(nocc, -1).T.conj())

    for n in range(1, nmax+1):
        vr = (
            - utils.einsum('ikjl,xkla->xija', oooo, vr) * sign
            + utils.einsum('ilba,xljb->xija', oovv, vr) * sign
            - utils.einsum('jalb,xilb->xija', ovov, vr) * sign * 2
            + utils.einsum('jalb,xlib->xija', ovov, vr) * sign
            + utils.einsum('jlba,xilb->xija', oovv, vr) * sign
            + utils.einsum('ija,xija->xija', eija, vr)
        )

        t[n] += utils.einsum('xija,yija->xy', vl, vr)

    return t


class ADCHelper(ip_radc2.ADCHelper):
    def build(self):
        self.eo, self.ev = self.e[self.o], self.e[self.v]
        self.co, self.cv = self.c[:,self.o], self.c[:,self.v]

        self.ovov = self.ao2mo(self.co, self.cv, self.co, self.cv)
        self.ooov = self.ao2mo(self.co, self.co, self.co, self.cv)
        self.oooo = self.ao2mo(self.co, self.co, self.co, self.co)
        self.oovv = self.ao2mo(self.co, self.co, self.cv, self.cv)

        self.eija = lib.direct_sum('i,j,a->ija', self.eo, self.eo, -self.ev)

        eiajb = lib.direct_sum('i,a,j,b->iajb', self.eo, -self.ev, self.eo, -self.ev)
        self.t2 = self.ovov / eiajb
        self.sign = 1
        self.guess_high_order = True

        self._to_unpack = ['t2', 'ovov', 'ooov', 'oooo', 'oovv', 'eija']

    get_matvec = get_matvec
    get_1h = get_1h
    get_moments = get_moments
