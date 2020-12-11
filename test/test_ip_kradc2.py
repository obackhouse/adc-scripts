import unittest
import numpy as np
from pyscf.pbc import gto, scf, tools
from adc import run


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cell = gto.Cell()
        cell.atom = 'He 0 0 0; He 1.685069 1.685069 1.685069'
        cell.basis = [[0, (1., 1.)], [0, (.5, 1.)]]
        cell.a = (np.ones((3,3)) - np.eye(3)) * 3.370137
        cell.unit = 'B'
        cell.verbose = 0
        cell.build()
        cls.cell = cell

    @classmethod
    def tearDownClass(cls):
        del cls.cell

    def test_ip_kradc2_a(self):
        mf = scf.KRHF(self.cell, exxdiv=None)
        mf = mf.density_fit()
        mf.kpts = self.cell.make_kpts([1,1,2])
        mf.run()

        scell = tools.super_cell(self.cell, [1,1,2])
        smf = scf.RHF(scell, exxdiv=mf.exxdiv)
        smf = smf.density_fit()
        smf.run()
        _cderi = np.vstack([x.copy() for x in smf.with_df.loop()])
        smf._eri = np.dot(_cderi.T, _cderi)
        del smf.with_df

        e1, v1 = run(mf, nroots=3, do_mp2=False)
        e2, v2 = run(smf, nroots=6, do_mp2=False)
        e1 = np.sort(np.concatenate(e1))
        e2 = np.sort(e2)
        self.assertAlmostEqual(e1[0], e2[0], 7)
        self.assertAlmostEqual(e1[1], e2[1], 7)
        self.assertAlmostEqual(e1[2], e2[2], 7)
        self.assertAlmostEqual(e1[3], e2[3], 7)
        self.assertAlmostEqual(e1[4], e2[4], 7)
        self.assertAlmostEqual(e1[5], e2[5], 7)

    def test_ip_kradc2_b(self):
        mf = scf.KRHF(self.cell, exxdiv='ewald')
        mf = mf.density_fit()
        mf.kpts = self.cell.make_kpts([1,2,2])
        mf.run()

        scell = tools.super_cell(self.cell, [1,2,2])
        smf = scf.RHF(scell, exxdiv=mf.exxdiv)
        smf = smf.density_fit()
        smf.run()
        _cderi = np.vstack([x.copy() for x in smf.with_df.loop()])
        smf._eri = np.dot(_cderi.T, _cderi)
        del smf.with_df

        e1, v1 = run(mf, nroots=3, do_mp2=False)
        e2, v2 = run(smf, nroots=6, do_mp2=False)
        e1 = np.sort(np.concatenate(e1))
        e2 = np.sort(e2)
        self.assertAlmostEqual(e1[0], e2[0], 6)
        self.assertAlmostEqual(e1[1], e2[1], 6)
        self.assertAlmostEqual(e1[2], e2[2], 6)
        self.assertAlmostEqual(e1[3], e2[3], 6)
        self.assertAlmostEqual(e1[4], e2[4], 6)
        self.assertAlmostEqual(e1[5], e2[5], 6)

    def test_ip_kradc2_c(self):
        mf = scf.KRHF(self.cell, exxdiv=None)
        mf.kpts = self.cell.make_kpts([1,2,2])
        mf.run()

        scell = tools.super_cell(self.cell, [1,2,2])
        smf = scf.RHF(scell, exxdiv=mf.exxdiv)
        smf.run()
        _cderi = np.vstack([x.copy() for x in smf.with_df.loop()])
        smf._eri = np.dot(_cderi.T, _cderi)
        del smf.with_df

        e1, v1 = run(mf, nroots=3, do_mp2=False)
        e2, v2 = run(smf, nroots=6, do_mp2=False)
        e1 = np.sort(np.concatenate(e1))
        e2 = np.sort(e2)
        self.assertAlmostEqual(e1[0], e2[0], 6)
        self.assertAlmostEqual(e1[1], e2[1], 6)
        self.assertAlmostEqual(e1[2], e2[2], 6)
        self.assertAlmostEqual(e1[3], e2[3], 6)
        self.assertAlmostEqual(e1[4], e2[4], 6)
        self.assertAlmostEqual(e1[5], e2[5], 6)


if __name__ == '__main__':
    print('IP-KRADC(2) tests')
    unittest.main()
