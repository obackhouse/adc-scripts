'''
ADC(3) for ionization potentials for restricted references.

This DF not like ADC(2) with N^3 memory, closer to the algorithm due to Sokolov et al.
'''

import numpy as np
from adc import utils, mpi_helper, ip_df_radc2
from pyscf import lib

def as1(x, axis=(1,3)):
    return x - x.swapaxes(*axis)

def as2(x, axis=(1,3)):
    return 2.0 * x - x.swapaxes(*axis)

def dot_along_tail(a, b):
    # iakb,jakb->ij or similar with MPI support

    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    assert a.shape[1] == b.shape[1]

    p0, p1 = mpi_helper.distr_blocks(a.shape[1])
    m = np.dot(a[:,p0:p1], b[:,p0:p1].T)

    mpi_helper.barrier()
    mpi_helper.allreduce_inplace(m)

    return m

def dot_along_tail2(a, b):
    # iakc,jbkc->iajb or similar with MPI support

    shape = (a.shape[0], a.shape[1], b.shape[0], b.shape[1])
    a = a.reshape(a.shape[0]*a.shape[1], -1)
    b = b.reshape(b.shape[0]*b.shape[1], -1)
    assert a.shape[1] == b.shape[1]

    p0, p1 = mpi_helper.distr_blocks(a.shape[1])
    m = np.dot(a[:,p0:p1], b[:,p0:p1].T)

    mpi_helper.barrier()
    mpi_helper.allreduce_inplace(m)

    return m.reshape(shape)

def get_matvec(helper):
    t1_2, t2, t2_2, ovov, ooov, oooo, oovv, ovvv, vvvv, eija = helper.unpack()
    nocc, nvir = helper.nocc, helper.nvir
    sign = helper.sign
    t2a = as1(t2)
    t2_2_a = as1(t2_2)

    h1  = np.diag(helper.eo)

    tmp1 = dot_along_tail(t2, as2(ovov)) * 0.5
    h1 += tmp1
    h1 += tmp1.T

    tmp1 = utils.einsum('ld,jild->ij', t1_2, as2(ooov, (0,2)))
    h1 += sign * tmp1
    h1 += sign * tmp1.T

    tmp1  = dot_along_tail(t2_2_a, as1(ovov)) * 0.5
    tmp1 += dot_along_tail(t2_2, ovov.swapaxes(1,3)) * 0.5
    tmp2  = lib.direct_sum('ijb,a->iajb', eija, -helper.ev)
    tmp1 -= dot_along_tail(as2(t2_2) * tmp2, t2) * 0.5
    h1 += sign * tmp1
    h1 += sign * tmp1.T

    tmp1  = dot_along_tail2(as2(t2), as2(t2))
    tmp2  = dot_along_tail(tmp1, ovov)
    tmp1  = dot_along_tail2(t2a, t2a)
    tmp1 += dot_along_tail2(t2, t2)
    tmp1 += dot_along_tail2(t2.swapaxes(1,3), t2.swapaxes(1,3))
    tmp2 -= dot_along_tail(tmp1.swapaxes(1,2), oovv)
    h1 += sign * tmp2
    h1 += sign * tmp2.T

    tmp1  = utils.einsum('jkab->jakb', helper._t2_oooo.copy())
    tmp1 += utils.einsum('jkab->jakb', helper._t2_vvvv.copy()) * 0.5
    tmp2  = dot_along_tail2(t2, oovv.swapaxes(1,2)) * -0.5
    tmp3  = dot_along_tail(t2, tmp1+tmp2)
    tmp2  = dot_along_tail2(t2a, oovv.swapaxes(1,2)) * -0.5
    tmp3 += dot_along_tail(t2a, 0.5*as1(tmp1)+tmp2)
    h1 += sign * tmp3
    h1 += sign * tmp3.T

    tmp1  = dot_along_tail(t2a, t2a)
    tmp1 += dot_along_tail(t2, t2) * 2.0
    for i in mpi_helper.distr_iter(range(nocc)):
        v = np.dot(helper.Loo[:,i].T, helper.Loo.reshape(-1, nocc*nocc)).reshape(nocc, nocc, nocc)
        v = 2.0 * v - v.swapaxes(0,2)
        h1[i] -= np.dot(v.reshape(nocc, -1), tmp1.ravel()) * sign * 0.5

    tmp1  = dot_along_tail(t2a.swapaxes(0,1), t2a.swapaxes(0,1))
    tmp1 += dot_along_tail(t2.swapaxes(0,1), t2.swapaxes(0,1)) * 2.0
    h1 += utils.einsum('jibc,bc->ij', oovv, tmp1) * sign 
    h1 -= utils.einsum('jcib,bc->ij', ovov, tmp1) * sign * 0.5

    tmp1  = dot_along_tail2(as2(t2), ovov)
    h1 += dot_along_tail(as2(t2), tmp1) * sign

    tmp1  = dot_along_tail2(t2.swapaxes(1,3), oovv.swapaxes(1,2))
    h1 -= dot_along_tail(t2.swapaxes(1,3), tmp1) * sign

    def matvec(y):
        y = np.asarray(y, order='C')
        r = np.zeros_like(y)

        yi = y[:nocc]
        ri = r[:nocc]
        yija = y[nocc:].reshape(nocc, nocc, nvir)
        rija = r[nocc:].reshape(nocc, nocc, nvir)
        yija_as = as1(yija, (0,1))

        ri   += np.dot(h1, yi)

        ri   += utils.einsum('kija,ija->k', as2(ooov, (1,2)), yija)
        rija += utils.einsum('k,kija->ija', yi, ooov)
        rija += eija * yija

        rija -= utils.einsum('ikjl,kla->ija', oooo, yija) * sign
        rija += utils.einsum('ilba,ljb->ija', oovv, yija) * sign
        rija -= utils.einsum('jalb,ilb->ija', ovov, as2(yija, (0,1))) * sign
        rija += utils.einsum('jlba,ilb->ija', oovv, yija) * sign

        tmp1  = utils.einsum('ibjc,jia->cab', t2a, as1(yija, (0,1))) * 0.25
        ri   += utils.einsum('cab,icab->i', tmp1, as1(ovvv)) * sign
        tmp1  = utils.einsum('kbjc,jka->cab', t2, yija)
        ri   += utils.einsum('cab,icab->i', tmp1, ovvv) * sign
        tmp1  = utils.einsum('i,ibac->cab', yi, ovvv)
        rija += utils.einsum('cab,jcib->ija', tmp1, t2) * sign

        tmp1  = utils.einsum('jalb,kja->blk', t2a, as1(yija, (0,1)))
        tmp1 += utils.einsum('jalb,kja->blk', t2, yija)
        ri   += utils.einsum('blk,iklb->i', tmp1, as1(ooov, (0,2))) * sign

        tmp1  = utils.einsum('jalb,kja->blk', t2, as1(yija, (0,1)))
        tmp1 += utils.einsum('jalb,kja->blk', t2a, yija)
        ri   += utils.einsum('blk,iklb->i', tmp1, ooov) * sign

        tmp1  = utils.einsum('jbla,jka->blk', t2, yija)
        ri   -= utils.einsum('blk,lkib->i', tmp1, ooov) * sign

        tmp1  = utils.einsum('i,iklb->kbl', yi, as1(ooov, (0,2)))
        rija += utils.einsum('kbl,jalb->kja', tmp1, t2) * sign

        tmp1  = utils.einsum('i,iklb->kbl', yi, ooov)
        rija += utils.einsum('kbl,jalb->kja', tmp1, t2a) * sign

        rija -= utils.einsum('i,ljib,kbla->kja', yi, ooov, t2) * sign

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

class ADCHelper(ip_df_radc2.ADCHelper):
    def build(self):
        self.eo, self.ev = self.e[self.o], self.e[self.v]
        self.co, self.cv = self.c[:,self.o], self.c[:,self.v]

        self.Loo = self.ao2mo(self.co, self.co)
        self.Lvv = self.ao2mo(self.cv, self.cv)
        self.Lov = self.ao2mo(self.co, self.cv)

        nocc, nvir = self.nocc, self.nvir
        self.ovov = np.dot(self.Lov.reshape(-1, nocc*nvir).T, self.Lov.reshape(-1, nocc*nvir)).reshape(nocc, nvir, nocc, nvir)
        self.oovv = np.dot(self.Loo.reshape(-1, nocc*nocc).T, self.Lvv.reshape(-1, nvir*nvir)).reshape(nocc, nocc, nvir, nvir)
        self.ooov = np.dot(self.Loo.reshape(-1, nocc*nocc).T, self.Lov.reshape(-1, nocc*nvir)).reshape(nocc, nocc, nocc, nvir)
        self.ovvv = np.dot(self.Lov.reshape(-1, nocc*nvir).T, self.Lvv.reshape(-1, nvir*nvir)).reshape(nocc, nvir, nvir, nvir)
        self.oooo = np.dot(self.Loo.reshape(-1, nocc*nocc).T, self.Loo.reshape(-1, nocc*nocc)).reshape(nocc, nocc, nocc, nocc)
        self.vvvv = np.dot(self.Lvv.reshape(-1, nvir*nvir).T, self.Lvv.reshape(-1, nvir*nvir)).reshape(nvir, nvir, nvir, nvir)

        self.eija = lib.direct_sum('i,j,a->ija', self.eo, self.eo, -self.ev)
        self.eajb = lib.direct_sum('a,j,b->ajb', -self.ev, self.eo, -self.ev)

        eia = self.eia = lib.direct_sum('i,a->ia', self.eo, -self.ev)
        eiajb = lib.direct_sum('ia,jb->iajb', eia, eia)
        self.t2 = self.ovov / eiajb

        t2a = self.t2 - self.t2.swapaxes(0,2).copy()
        self.t1_2  = utils.einsum('kdac,ickd->ia', self.ovvv, self.t2+t2a*0.5)
        self.t1_2 -= utils.einsum('kcad,ickd->ia', self.ovvv, t2a) * 0.5
        self.t1_2 -= utils.einsum('kilc,kalc->ia', self.ooov, self.t2+t2a*0.5)
        self.t1_2 -= utils.einsum('likc,lakc->ia', self.ooov, t2a) * 0.5
        self.t1_2 /= eia

        self._t2_oooo = np.zeros((nocc, nocc, nvir, nvir))
        self._t2_vvvv = np.zeros((nocc, nocc, nvir, nvir))
        for i in mpi_helper.distr_iter(range(nocc)):
            v = np.dot(self.Loo[:,i].T, self.Loo.reshape(-1, nocc*nocc)).reshape(nocc, nocc, nocc)
            self._t2_oooo += np.tensordot(v, self.t2[i], axes=((1,),(1,)))
        for a in mpi_helper.distr_iter(range(nvir)):
            v = np.dot(self.Lvv[:,a].T, self.Lvv.reshape(-1, nvir*nvir)).reshape(nvir, nvir, nvir)
            self._t2_vvvv += np.tensordot(self.t2[:,a], v, axes=((2,),(2,)))

        self.t2_2  = utils.einsum('ijab->iajb', self._t2_oooo.copy())
        self.t2_2 += utils.einsum('ijab->iajb', self._t2_vvvv.copy())
        self.t2_2 += utils.einsum('kcjb,iakc->iajb', self.ovov, self.t2+t2a)
        self.t2_2 -= utils.einsum('kjbc,iakc->iajb', self.oovv, self.t2)
        self.t2_2 -= utils.einsum('kibc,kajc->iajb', self.oovv, self.t2)
        self.t2_2 -= utils.einsum('kjac,ickb->iajb', self.oovv, self.t2)
        self.t2_2 += utils.einsum('kcia,kcjb->iajb', self.ovov, self.t2+t2a)
        self.t2_2 -= utils.einsum('kiac,kcjb->iajb', self.oovv, self.t2)
        self.t2_2 /= eiajb

        self.sign = 1
        self.guess_high_order = True

        self._to_unpack = ['t1_2', 't2', 't2_2', 'ovov', 'ooov', 'oooo', 'oovv', 'ovvv', 'vvvv', 'eija']

    get_matvec = get_matvec
