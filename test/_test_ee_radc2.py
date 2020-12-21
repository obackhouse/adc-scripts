import unittest
import numpy as np
from pyscf import gto, scf
from adc import run


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz', verbose=0)
        mol = gto.M(atom='H 0 0 0; Li 0 0 1', basis='cc-pvdz', verbose=0)
        cls.mf = scf.RHF(mol)
        cls.mf.run(conv_tol=1e-12)

    @classmethod
    def tearDownClass(cls):
        del cls.mf

    def test_ee_radc2(self):
        e, v, e_mp2 = run(self.mf, which='ee', nroots=5, tol=1e-12, do_mp2=True)
        #self.assertAlmostEqual(e_mp2, -0.20905685, 7)
        print(e)
        import adcc
        adcc.run_adc(self.mf, method='adc2', n_states=5)
        #self.assertAlmostEqual(e[0],   0.15307444, 7) 
        #self.assertAlmostEqual(e[1],   0.23140882, 7) 
        #self.assertAlmostEqual(e[2],   0.67633060, 7) 
        #self.assertAlmostEqual(e[3],   0.78837830, 7) 
        #self.assertAlmostEqual(e[4],   0.84460837, 7)


if __name__ == '__main__':
    print('EE-RADC(2) tests')
    unittest.main()
