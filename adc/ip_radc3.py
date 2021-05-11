'''
ADC(3) for ionization potentials for restricted references.
'''

import numpy as np
from adc import utils, mpi_helper, ip_radc2
from pyscf import lib

def as1(x, axis=(1,3)):
    return x - x.swapaxes(*axis)

def as2(x, axis=(1,3)):
    return 2.0 * x - x.swapaxes(*axis)

def dot_along_tail(a, b):
    # shortcut to mpi_helper.einsum('iakb,jakb->ij', a, b)
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    return mpi_helper.dot(a, b.T)

def dot_along_tail2(a, b):
    # shortcut to mpi_helper.einsum('iakc,jbkc->iajb', a, b)
    shape = (a.shape[0], a.shape[1], b.shape[0], b.shape[1])
    a = a.reshape(a.shape[0]*a.shape[1], -1)
    b = b.reshape(b.shape[0]*b.shape[1], -1)
    return mpi_helper.dot(a, b.T).reshape(shape)

def get_1h(helper):
    t1_2, t2, t2_2, ovov, ooov, oooo, oovv, ovvv, vvvv, eija = helper.unpack()
    nocc, nvir = helper.nocc, helper.nvir
    sign = helper.sign
    t2a = as1(t2)
    t2_2_a = as1(t2_2)

    h1  = np.diag(helper.eo)

    tmp1 = utils.einsum('iakb,jakb->ij', t2, as2(ovov)) * 0.5
    h1 += tmp1
    h1 += tmp1.T

    tmp1 = utils.einsum('ld,jild->ij', t1_2, as2(ooov, (0,2)))
    h1 += sign * tmp1
    h1 += sign * tmp1.T

    tmp1  = dot_along_tail(t2_2_a, as1(ovov)) * 0.5
    tmp1 += dot_along_tail(t2_2, ovov.swapaxes(1,3)) * 0.5
    tmp2  = lib.direct_sum('ijb,a->iajb', eija, -helper.ev)
    tmp1 -= utils.einsum('iakb,jakb->ij', as2(t2_2) * tmp2, t2) * 0.5
    h1 += sign * tmp1
    h1 += sign * tmp1.T

    tmp1  = utils.einsum('lckb,jalc->jakb', as2(t2), as2(t2))
    tmp2  = utils.einsum('jakb,iakb->ij', tmp1, ovov)
    tmp1  = utils.einsum('lckb,jalc->jkab', t2a, t2a)
    tmp1 += utils.einsum('lckb,jalc->jkab', t2, t2)
    tmp1 += utils.einsum('lakc,jclb->jkab', t2, t2)
    tmp2 -= utils.einsum('ikab,jkab->ij', tmp1, oovv)
    h1 += sign * tmp2
    h1 += sign * tmp2.T

    tmp1  = utils.einsum('jkab->jakb', helper._t2_oooo.copy())
    tmp1 += utils.einsum('jkab->jakb', helper._t2_vvvv.copy()) * 0.5
    tmp2  = utils.einsum('jalc,klbc->jakb', t2, oovv) * -0.5
    tmp3  = utils.einsum('iakb,jakb->ij', t2, tmp1+tmp2)
    tmp2  = utils.einsum('jalc,klbc->jakb', t2a, oovv) * -0.5
    tmp3 += utils.einsum('iakb,jakb->ij', t2a, 0.5*as1(tmp1)+tmp2)
    h1 += sign * tmp3
    h1 += sign * tmp3.T

    tmp1  = utils.einsum('iakb,ialb->kl', t2a, t2a)
    tmp1 += utils.einsum('iakb,ialb->kl', t2, t2) * 2.0
    h1 -= utils.einsum('ijkl,kl->ij', as2(oooo), tmp1) * sign * 0.5

    tmp1  = utils.einsum('iakc,iakb->bc', t2a, t2a)
    tmp1 += utils.einsum('iakc,iakb->bc', t2, t2) * 2.0
    h1 += utils.einsum('jibc,bc->ij', oovv, tmp1) * sign 
    h1 -= utils.einsum('jcib,bc->ij', ovov, tmp1) * sign * 0.5

    tmp1 = utils.einsum('jalc,kblc->jakb', as2(t2), ovov)
    h1 += utils.einsum('iakb,jakb->ij', as2(t2), tmp1) * sign

    tmp1 = utils.einsum('jcla,klbc->jakb', t2, oovv)
    h1 -= utils.einsum('ibka,jakb->ij', t2, tmp1) * sign

    return h1


def get_matvec(helper):
    t1_2, t2, t2_2, ovov, ooov, oooo, oovv, ovvv, vvvv, eija = helper.unpack()
    nocc, nvir = helper.nocc, helper.nvir
    sign = helper.sign
    t2a = as1(t2)

    h1 = get_1h(helper)

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

    #FIXME: is this broken? not sure it is correct
    if helper.guess_high_order:
        # According to A. Sokolov these might actually make things worse
        # See https://github.com/pyscf/pyscf/commit/994e325159866bc74319418033db270a6b6a9d57#r45037621
        diag_ija = diag[nocc:].reshape(nocc, nocc, nvir)
        diag_ija -= utils.einsum('iijj->ij', oooo)[:,:,None] * sign
        diag_ija += utils.einsum('jjaa->ja', oovv)[None,:,:] * sign
        diag_ija += utils.einsum('iiaa->ia', oovv)[:,None,:] * sign

    return matvec, diag


def get_moments(helper, nmax):
    t1_2, t2, t2_2, ovov, ooov, oooo, oovv, ovvv, vvvv, eija = helper.unpack()
    nocc, nvir = helper.nocc, helper.nvir
    sign = helper.sign
    t2a = as2(t2)

    vl  = as2(ooov, axis=(1,2))
    vl += utils.einsum('jbic,kcba->kija', t2a, ovvv) * sign
    vl += utils.einsum('jalb,kilb->kija', t2a, ooov) * sign * 2
    vl -= utils.einsum('jalb,likb->kija', t2a, ooov) * sign
    vl -= utils.einsum('ialb,kjlb->kija', t2a, ooov) * sign
    vl -= utils.einsum('ibla,jlkb->kija', t2a, ooov) * sign

    vr  = ooov.copy()
    vr += utils.einsum('ibjc,kbca->kija', t2,  ovvv) * sign
    vr -= utils.einsum('ibla,jlkb->kija', t2,  ooov) * sign
    vr += utils.einsum('jalb,kilb->kija', t2a, ooov) * sign
    vr -= utils.einsum('jalb,ilkb->kija', t2,  ooov) * sign

    t = np.zeros((nmax+1, nocc, nocc), dtype=ovov.dtype)
    t[0] = np.dot(vl.reshape(nocc, -1), vr.reshape(nocc, -1).T.conj())

    for n in range(1, nmax+1):
        vr = (
            - utils.einsum('ikjl,xkla->xija', oooo, vr) * sign
            + utils.einsum('ilba,xljb->xija', oovv, vr) * sign
            - utils.einsum('jalb,xilb->xija', ovov, as2(vr, (1,2))) * sign
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
        self.ovvv = self.ao2mo(self.co, self.cv, self.cv, self.cv)
        self.vvvv = self.ao2mo(self.cv, self.cv, self.cv, self.cv)

        self.eija = lib.direct_sum('i,j,a->ija', self.eo, self.eo, -self.ev)

        eia = lib.direct_sum('i,a->ia', self.eo, -self.ev)
        eiajb = lib.direct_sum('ia,jb->iajb', eia, eia)
        self.t2 = self.ovov / eiajb

        self._t2_oooo = np.tensordot(self.oooo, self.t2, axes=((0,2),(0,2)))
        self._t2_vvvv = np.tensordot(self.t2, self.vvvv, axes=((1,3),(0,2)))

        t2a = self.t2 - self.t2.swapaxes(0,2).copy()
        self.t1_2  = utils.einsum('kdac,ickd->ia', self.ovvv, self.t2+t2a)
        self.t1_2 -= utils.einsum('kilc,kalc->ia', self.ooov, self.t2+t2a)
        self.t1_2 /= eia

        self.t2_2  = utils.einsum('ijab->iajb', self._t2_oooo.copy())
        self.t2_2 += utils.einsum('ijab->iajb', self._t2_vvvv.copy())
        self.t2_2 += utils.einsum('kcjb,iakc->iajb', self.ovov, self.t2+t2a)
        self.t2_2 -= utils.einsum('kjcb,iakc->iajb', self.oovv, self.t2)
        self.t2_2 -= utils.einsum('kicb,kajc->iajb', self.oovv, self.t2)
        self.t2_2 -= utils.einsum('kjac,ickb->iajb', self.oovv, self.t2)
        self.t2_2 += utils.einsum('kcia,kcjb->iajb', self.ovov, self.t2+t2a)
        self.t2_2 -= utils.einsum('kica,kcjb->iajb', self.oovv, self.t2)
        self.t2_2 /= eiajb

        self.sign = 1
        self.guess_high_order = True

        self._to_unpack = ['t1_2', 't2', 't2_2', 'ovov', 'ooov', 'oooo', 'oovv', 'ovvv', 'vvvv', 'eija']

    get_matvec = get_matvec
    get_1h = get_1h
    get_moments = get_moments
