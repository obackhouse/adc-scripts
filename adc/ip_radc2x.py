'''
ADC(2)-x for ionization potentials for restricted references.
'''

import numpy as np
from adc import utils, mpi_helper, ip_radc2
from pyscf import lib


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

    h1  = np.dot(t2_block, ovov_as_block.T) * 0.5

    mpi_helper.barrier()
    mpi_helper.allreduce_inplace(h1)

    h1 += h1.T
    h1 += np.diag(helper.eo)

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

        mpi_helper.barrier()
        mpi_helper.allreduce_inplace(ri)
        mpi_helper.allreduce_inplace(rija)

        ri += np.dot(h1, yi)

        yija = yija.reshape(nocc, nocc, nvir)
        yija_as = 2.0 * yija - yija.swapaxes(0,1)

        rija -= utils.einsum('ikjl,kla->ija', oooo, yija).ravel()    * sign
        rija -= utils.einsum('jalb,ilb->ija', ovov, yija_as).ravel() * sign
        rija += utils.einsum('ilba,ljb->ija', oovv, yija).ravel()    * sign
        rija += utils.einsum('jlba,ilb->ija', oovv, yija).ravel()    * sign

        return r

    diag = np.concatenate([np.diag(h1), eija.ravel()])
    diag_ija = diag[nocc:]

    p_oooo = oooo.swapaxes(1,2).reshape(nocc*nocc, nocc*nocc)
    p_ovov = oovv.swapaxes(1,2).reshape(nocc*nvir, nocc*nvir)

    tmp  = np.zeros((nvir, nocc*nocc))
    tmp += np.diag(p_oooo)[None]
    tmp  = np.transpose(tmp)
    diag_ija -= tmp.ravel() * sign

    tmp  = np.zeros((nocc, nocc, nvir))
    tmp += np.diag(p_ovov).reshape(-1, nocc, nvir)
    diag_ija += tmp.ravel() * sign

    tmp  = np.zeros((nocc, nocc, nvir))
    tmp += np.diag(p_ovov).reshape(-1, nocc, nvir)
    tmp  = tmp.swapaxes(0,1)
    diag_ija += tmp.ravel() * sign

    return matvec, diag

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

        self._to_unpack = ['t2', 'ovov', 'ooov', 'oooo', 'oovv', 'eija']

    get_matvec = get_matvec
