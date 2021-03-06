import unittest
import numpy as np
from pyscf import gto, scf
from adc import run, methods, ea_radc2x, utils


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

    def test_ea_radc2x_moments(self):
        mol = gto.M(atom='Li 0 0 0; H 0 0 1.64', basis='6-31g', verbose=0)
        mf = scf.RHF(mol)
        mf.run(conv_tol=1e-12)
        helper = ea_radc2x.ADCHelper(mf)
        nmax = 6
        m = helper.get_dense_array()
        t = helper.get_moments(nmax)
        
        def matpow(x, n):
            if n == 0: return np.eye(x.shape[0], dtype=x.dtype)
            elif n == 1: return x
            else: return np.linalg.multi_dot([x,]*n)

        for n in range(nmax+1):
            t_ref = np.dot(np.dot(
                    m[:helper.nocc, helper.nocc:],
                    matpow(m[helper.nocc:, helper.nocc:], n)),
                    m[helper.nocc:, :helper.nocc],
            )
            self.assertAlmostEqual(np.max(np.absolute(t[n]-t_ref)), 0.0, 8)


if __name__ == '__main__':
    print('EA-RADC(2)-x tests')
    unittest.main()
