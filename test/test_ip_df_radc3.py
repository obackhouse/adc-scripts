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

    def test_ip_df_radc3(self):
        # [0.44743838 0.55574549 0.63750722 1.09105829 1.14208418]
        # [0.15472896 0.23249965 0.41121217 0.45801825 0.56210971]
        e, v, conv, e_mp2 = run(self.mf, method='3', nroots=5, tol=1e-12, do_mp2=True)
        self.assertTrue(all(conv))
        self.assertAlmostEqual(e_mp2, -0.20905685, 5)
        self.assertAlmostEqual(e[0],   0.44743838, 5) 
        self.assertAlmostEqual(e[1],   0.55574549, 5) 
        self.assertAlmostEqual(e[2],   0.63750722, 5) 
        self.assertAlmostEqual(e[3],   1.09105829, 5) 
        self.assertAlmostEqual(e[4],   1.14208418, 5)


if __name__ == '__main__':
    print('IP-DF-RADC(3) tests')
    unittest.main()
