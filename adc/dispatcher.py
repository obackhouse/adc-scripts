'''
Dispatcher for ADC methods.
'''

import numpy as np
from adc import utils
from adc import methods
from pyscf import lib, scf


def get_picker(koopmans=False, real_system=False, guess=None):
    if not koopmans:
        def pick(w, v, nroots, envs):
            w, v, idx = lib.linalg_helper.pick_real_eigs(w, v, nroots, envs)
            mask = np.argsort(np.absolute(w))
            return w[mask], v[:,mask], idx

    else:
        assert guess is not None

        def pick(w, v, nroots, envs):
            x0 = lib.linalg_helper._gen_x0(envs['v'], envs['xs'])
            s = np.dot(np.asarray(guess).conj(), np.asarray(x0).T)
            snorm = utils.einsum('pi,pi->i', s.conj(), s)
            idx = np.argsort(-snorm)[:nroots]
            return lib.linalg_helper._eigs_cmplx2real(w, v, idx, real_system)

    return pick


def load_helper(mf, method='2', which='ip'):
    hf_type = 'r' if isinstance(mf, scf.hf.RHF) else 'u'
    periodicity = 'k' if hasattr(mf, 'kpts') else ''
    integral_type = 'df_' if getattr(mf, 'with_df', False) and periodicity == '' else ''
    method_name = '%s_%s%s%sadc%s' % (which, integral_type, periodicity, hf_type, method)

    try:
        module = getattr(methods, method_name)
    except KeyError:
        raise NotImplementedError(method_name)

    return module.ADCHelper


def run(mf, helper=None, method='2', which='ip', nroots=5, tol=1e-9,
        maxiter=100, maxspace=12, do_mp2=False, koopmans=False, verbose=False):
    ''' Runs the ADC method.

    Arguments:
        mf : scf.HF
            Mean-field method from pyscf, must be converged.
        helper : ADCHelper
            Specify the ADCHelper class, if None then it is handled
            automatically by the attributes mf and which.
        method : str
            One of '2', '2x', '3'
        which : str
            One of 'ip', 'ea' or 'ee'.
        nroots : int
            Number of states to solver for.
        tol : float
            Convergence tolerance for Davidson method.
        do_mp2 : bool
            Whether to compute the MP2 energy.
        koopmans : bool
            Target only quasiparticle-like states

    Returns:
        e : ndarray
            Excitation energies, may be in a nested list if an unrestricted
            reference or PBC is used.
        v : ndarray
            Eigenvectors, structured as above.
        mp2 : float
            MP2 correlation energy, if do_mp2 is True.
    '''

    if hasattr(mf, 'kpts'):
        return _run_pbc(mf, helper, method, which, nroots, tol, maxiter, maxspace, do_mp2, koopmans)

    if helper is None:
        helper = load_helper(mf, method=method, which=which)
    if callable(helper):
        helper = helper(mf)

    matvec, diag = helper.get_matvec()
    matvecs = lambda xs: [matvec(x) for x in xs]
    guesses = helper.get_guesses(diag, nroots, koopmans=koopmans)
    pick = get_picker(koopmans=koopmans, real_system=True, guess=guesses)
    kwargs = dict(tol=tol, nroots=nroots, pick=pick, max_cycle=maxiter,
                  max_space=maxspace, verbose=9 if verbose else 0)

    conv, e, v = lib.davidson_nosym1(matvecs, guesses, diag, **kwargs)

    if which == 'ip':
        e = utils.nested_apply(e, lambda x: -x)

    if do_mp2:
        mp2 = helper.mp2()
        if which == 'ea':
            mp2 *= -1
        return e, v, conv, mp2
    else:
        return e, v, conv


def _run_pbc(mf, helper=None, method='2', which='ip', nroots=5, tol=1e-12,
             maxiter=100, maxspace=12, do_mp2=False, koopmans=False, verbose=False):
    if helper is None:
        helper = load_helper(mf, method=method, which=which)
    if callable(helper):
        helper = helper(mf)

    es = []
    vs = []
    mp2 = []
    convs = []

    for ki in range(helper.nkpts):
        matvec, diag = helper.get_matvec(ki)
        matvecs = lambda xs: [matvec(x) for x in xs]
        guesses = helper.get_guesses(ki, diag, nroots, koopmans=koopmans)
        pick = get_picker(koopmans=koopmans, real_system=False, guess=guesses)
        kwargs = dict(tol=tol, nroots=nroots, pick=pick, max_cycle=maxiter,
                      max_space=maxspace, verbose=9 if verbose else 0)

        conv, e, v = lib.davidson_nosym1(matvecs, guesses, diag, **kwargs)

        if which == 'ip':
            e = utils.nested_apply(e, lambda x: -x)

        es.append(e)
        vs.append(v)
        convs.append(conv)

    if do_mp2:
        mp2 = helper.mp2()
        if which == 'ea':
            mp2 *= -1
        return es, vs, convs, mp2
    else:
        return es, vs, convs
