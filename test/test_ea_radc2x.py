import unittest
import numpy as np
from pyscf import gto, scf
from adc import run, methods


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz', verbose=0)
        cls.mf = scf.RHF(mol)
        cls.mf.run(conv_tol=1e-12)

    @classmethod
    def tearDownClass(cls):
        del cls.mf

    def test_ea_radc2x(self):
        e, v, conv, e_mp2 = run(self.mf, method='2x', which='ea', nroots=5, tol=1e-12, do_mp2=True)
        self.assertTrue(all(conv))
        self.assertAlmostEqual(e_mp2, -0.20905685, 7)
        self.assertAlmostEqual(e[0],   0.14962281, 7) 
        self.assertAlmostEqual(e[1],   0.22781683, 7) 
        self.assertAlmostEqual(e[2],   0.41120578, 7) 
        self.assertAlmostEqual(e[3],   0.45801734, 7) 
        self.assertAlmostEqual(e[4],   0.56211438, 7)


if __name__ == '__main__':
    print('EA-RADC(2)-x tests')
    unittest.main()
