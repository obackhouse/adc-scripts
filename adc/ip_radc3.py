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

def get_matvec(helper):
    t1_2, t2, t2_2, ovov, ooov, oooo, oovv, ovoo, ovvv, vvvv, eija = helper.unpack()
    nocc, nvir = helper.nocc, helper.nvir
    sign = helper.sign
    t2a = as1(t2)
    t2_2_a = as1(t2_2)

    h1  = np.diag(helper.eo)

    tmp1 = utils.einsum('iakb,jakb->ij', t2, as2(ovov)) * 0.5
    h1 += tmp1
    h1 += tmp1.T

    tmp1 = utils.einsum('ld,ldji->ij', t1_2, as2(ovoo, (0,2)))
    h1 += sign * tmp1
    h1 += sign * tmp1.T

    tmp1  = utils.einsum('iakb,jakb->ij', t2_2_a, as1(ovov)) * 0.5 
    tmp1 += utils.einsum('iakb,jbka->ij', t2_2, ovov) * 0.5
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
    tmp1  = as1(tmp1) * 0.5
    tmp3 += utils.einsum('iakb,jakb->ij', t2a, tmp1+tmp2)
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
        ri   += utils.einsum('blk,lbik->i', tmp1, as1(ovoo, (0,2))) * sign

        tmp1  = utils.einsum('jalb,kja->blk', t2, as1(yija, (0,1)))
        tmp1 += utils.einsum('jalb,kja->blk', t2a, yija)
        ri   += utils.einsum('blk,lbik->i', tmp1, ovoo) * sign

        tmp1  = utils.einsum('jbla,jka->blk', t2, yija)
        ri   -= utils.einsum('blk,iblk->i', tmp1, ovoo) * sign

        tmp1  = utils.einsum('i,lbik->kbl', yi, as1(ovoo, (0,2)))
        rija += utils.einsum('kbl,jalb->kja', tmp1, t2) * sign

        tmp1  = utils.einsum('i,lbik->kbl', yi, ovoo)
        rija += utils.einsum('kbl,jalb->kja', tmp1, t2a) * sign

        rija -= utils.einsum('i,iblj,kbla->kja', yi, ovoo, t2) * sign

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

class ADCHelper(ip_radc2.ADCHelper):
    def build(self):
        self.eo, self.ev = self.e[self.o], self.e[self.v]
        self.co, self.cv = self.c[:,self.o], self.c[:,self.v]

        self.ovov = self.ao2mo(self.co, self.cv, self.co, self.cv)
        self.ooov = self.ao2mo(self.co, self.co, self.co, self.cv)
        self.oooo = self.ao2mo(self.co, self.co, self.co, self.co)
        self.oovv = self.ao2mo(self.co, self.co, self.cv, self.cv)
        self.ovoo = self.ao2mo(self.co, self.cv, self.co, self.co)
        self.ovvv = self.ao2mo(self.co, self.cv, self.cv, self.cv)
        self.vvvv = self.ao2mo(self.cv, self.cv, self.cv, self.cv)

        self.eija = lib.direct_sum('i,j,a->ija', self.eo, self.eo, -self.ev)

        eia = lib.direct_sum('i,a->ia', self.eo, -self.ev)
        eiajb = lib.direct_sum('ia,jb->iajb', eia, eia)
        self.t2 = self.ovov / eiajb

        self._t2_oooo = np.tensordot(self.oooo, self.t2, axes=((0,2),(0,2)))
        self._t2_vvvv = np.tensordot(self.t2, self.vvvv, axes=((1,3),(0,2)))

        t2a = self.t2 - self.t2.swapaxes(0,2).copy()
        self.t1_2  = utils.einsum('kdac,ickd->ia', self.ovvv, self.t2+t2a*0.5)
        self.t1_2 -= utils.einsum('kcad,ickd->ia', self.ovvv, t2a) * 0.5
        self.t1_2 -= utils.einsum('lcki,kalc->ia', self.ovoo, self.t2+t2a*0.5)
        self.t1_2 -= utils.einsum('kcli,lakc->ia', self.ovoo, t2a) * 0.5
        self.t1_2 /= eia

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

        self._to_unpack = ['t1_2', 't2', 't2_2', 'ovov', 'ooov', 'oooo', 'oovv', 'ovoo', 'ovvv', 'vvvv', 'eija']

    get_matvec = get_matvec
