import unittest
import numpy as np
from pyscf import gto, scf
from adc2 import run


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz', verbose=0)
        cls.mf = scf.RHF(mol)
        cls.mf = cls.mf.density_fit(auxbasis='cc-pv5z-ri')
        cls.mf.run(conv_tol=1e-12)

    @classmethod
    def tearDownClass(cls):
        del cls.mf

    def test_ea_radc2(self):
        e, v, e_mp2 = run(self.mf, which='ea', nroots=5, do_mp2=True)
        self.assertAlmostEqual(e_mp2, -0.20905685, 5)
        self.assertAlmostEqual(e[0],   0.15307444, 5) 
        self.assertAlmostEqual(e[1],   0.23140882, 5) 
        self.assertAlmostEqual(e[2],   0.67633060, 5) 
        self.assertAlmostEqual(e[3],   0.78837830, 5) 
        self.assertAlmostEqual(e[4],   0.84460837, 5)


if __name__ == '__main__':
    print('EA-DF-RADC(2) tests')
    unittest.main()
