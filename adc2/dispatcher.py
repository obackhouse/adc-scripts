'''
Dispatcher for ADC(2) methods.
'''

import numpy as np
from adc2 import utils
from adc2 import ip_radc2, ea_radc2
from adc2 import ip_df_radc2, ea_df_radc2
from pyscf import lib, scf


def pick(w, v, nroots, callback):
    w, v, idx = lib.linalg_helper.pick_real_eigs(w, v, nroots, callback)
    mask = np.argsort(np.absolute(w))
    return w[mask], v[:,mask], idx

def load_helper(mf, which='ip'):
    integral_type = 'df_' if hasattr(mf, 'with_df') else ''
    hf_type = 'r' if isinstance(mf, scf.hf.RHF) else 'u'
    method_name = '%s_%s%sadc2' % (which, integral_type, hf_type)

    try:
        module = globals()[method_name]
    except KeyError:
        raise NotImplementedError(method_name)

    return module.ADCHelper

def run(mf, which='ip', nroots=5, tol=1e-12, maxiter=100, maxspace=12, do_mp2=False):
    ''' Runs the ADC(2) method.

    Arguments:
        mf : scf.HF
            Mean-field method from pyscf, must be converged.
        which : str
            One of 'ip', 'ea' or 'ee'.
        nroots : int
            Number of states to solver for.
        tol : float
            Convergence tolerance for Davidson method.
        do_mp2 : bool
            Whether to compute the MP2 energy.

    Returns:
        e : ndarray
            Excitation energies, may be in a nested list if an unrestricted
            reference or PBC is used.
        v : ndarray
            Eigenvectors, structured as above.
        mp2 : float
            MP2 correlation energy, if do_mp2 is True.
    '''

    ADCHelper = load_helper(mf, which=which)
    helper = ADCHelper(mf)

    matvec, diag = helper.get_matvec()
    guesses = helper.get_guesses(diag, nroots)
    kwargs = dict(tol=tol, nroots=nroots, pick=pick, max_cycle=maxiter, max_space=maxspace)

    #e, v = lib.davidson_nosym(matvec, guesses, diag, **kwargs)
    conv, e, v = lib.davidson_nosym1(lambda xs: [matvec(x) for x in xs], guesses, diag, **kwargs)

    if which == 'ip':
        e = utils.nested_apply(e, lambda x: -x)

    if do_mp2:
        mp2 = helper.mp2()
        if which == 'ea':
            mp2 *= -1
        return e, v, mp2
    else:
        return e, v
