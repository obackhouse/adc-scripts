import unittest
import numpy as np
from pyscf import gto, scf
from adc import run, methods


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz', verbose=0)
        cls.mf = scf.RHF(mol)
        cls.mf = cls.mf.density_fit(auxbasis='aug-cc-pvqz-ri')
        cls.mf.run(conv_tol=1e-12)

    @classmethod
    def tearDownClass(cls):
        del cls.mf

    def test_ea_df_radc3(self):
        e, v, conv, e_mp2 = run(self.mf, method='3', which='ea', nroots=5, tol=1e-12, do_mp2=True)
        self.assertTrue(all(conv))
        self.assertAlmostEqual(e_mp2, -0.20905685, 5)
        self.assertAlmostEqual(e[0],   0.15472896, 5) 
        self.assertAlmostEqual(e[1],   0.23249965, 5) 
        self.assertAlmostEqual(e[2],   0.41121217, 5) 
        self.assertAlmostEqual(e[3],   0.45801825, 5) 
        self.assertAlmostEqual(e[4],   0.56210971, 5)


if __name__ == '__main__':
    print('EA-DF-RADC(3) tests')
    unittest.main()
