'''
MPI helper functions
Mostly adapted from pyscf and mpi4pyscf
'''

import numpy as np
import itertools
from pyscf import lib

try:
    from mpi4py import MPI as mpi
    comm = mpi.COMM_WORLD
    size, rank = comm.Get_size(), comm.Get_rank()
except:
    mpi = comm = None
    size, rank = 1, 0

INT_MAX = 2147483647
BLKSIZE = INT_MAX // 32 + 1


def check_for_mpi(function):
    def wrapper(*args, **kwargs):
        if size == 1:
            return args[0] if len(args) else None
        else:
            return function(*args, **kwargs)

    return wrapper

def as_acceptable_array(function):
    def wrapper(*args, **kwargs):
        buf = args[0]
        is_array = isinstance(buf, np.ndarray)

        buf = np.asarray(buf, order='C', dtype=buf.dtype.char)
        out = function(*args, **kwargs)

        if not is_array:
            out = out.ravel()[0]

        return out

    return wrapper

@check_for_mpi
@as_acceptable_array
def bcast(buf, root=0):
    shape, dtype = comm.bcast((buf.shape, buf.dtype.char))

    if rank != root:
        buf = np.empty(shape, dtype=dtype)

    seg = np.ndarray(buf.size, dtype=buf.dtype, buffer=buf)
    for p0, p1 in lib.prange(0, buf.size, BLKSIZE):
        comm.Bcast(seg[p0:p1], root)

    return buf

@check_for_mpi
@as_acceptable_array
def allreduce(sendbuf, root=0, op=getattr(mpi, 'SUM', None)):
    shape, dtype = comm.bcast((sendbuf.shape, sendbuf.dtype.char))
    assert sendbuf.shape == shape and sendbuf.dtype.char == dtype

    recvbuf = np.zeros_like(sendbuf)
    send_seg = np.ndarray(sendbuf.size, dtype=dtype, buffer=sendbuf)
    recv_seg = np.ndarray(recvbuf.size, dtype=dtype, buffer=recvbuf)
    for p0, p1 in lib.prange(0, sendbuf.size, BLKSIZE):
        comm.Allreduce(send_seg[p0:p1], recv_seg[p0:p1], op)

    return recvbuf

@check_for_mpi
@as_acceptable_array
def allreduce_inplace(sendbuf):
    from pyscf.pbc.mpitools.mpi_helper import safeAllreduceInPlace
    safeAllreduceInPlace(comm, sendbuf)
    return sendbuf

@check_for_mpi
def barrier():
    comm.Barrier()

@check_for_mpi
def distr_iter(iterable):
    iterable = list(iterable)
    for i in iterable[rank::size]:
        yield i

def distr_blocks(block):
    start = rank * (block // size)
    end = (rank+1) * (block // size)
    if (rank+1) == size:
        end = max(end, block)
    return start, end

def allclose(a):
    # numpy.allclose between MPI processes

    a_in = a.copy()
    a = allreduce(a_in) / size
    
    return np.allclose(a_in, a)

@check_for_mpi
def correct_vector_phase(v, full_check=False, extra=None):
    # vectors may be out by a sign between MPI processes

    if full_check:
        el = v
    else:
        el = v.ravel()[[0]]

    barrier()
    root_el = bcast(el)

    if np.linalg.norm(root_el + el) < np.linalg.norm(root_el - el):
        v *= -1
        if extra:
            extra = [-x for x in extra]

    if extra is None:
        return v
    else:
        return (v,) + tuple(extra)

@check_for_mpi
def mean(m):
    # take the mean of an array on each MPI process to gurantee determinism

    barrier()
    allreduce_inplace(m)
    m /= size

    return m

def dot(a, b):
    # numpy.dot with MPI support

    squeeze_left = squeeze_right = False
    if a.ndim == 1:
        squeeze_left = True
        a = a.reshape(1, -1)
    elif a.ndim > 2:
        raise ValueError
    if b.ndim == 1:
        squeeze_right = True
        b = b.reshape(-1, 1)
    elif b.ndim > 2:
        raise ValueError

    assert a.shape[1] == b.shape[0]
    dtype = np.result_type(a.dtype, b.dtype)
    axes = (a.shape[0], a.shape[1], b.shape[0])
    choice = np.argmax(axes)

    p0, p1 = distr_blocks(axes[choice])
    res = np.zeros((a.shape[0], b.shape[1]), dtype=dtype)

    if choice == 0:
        res[p0:p1] = np.dot(a[p0:p1], b, out=res[p0:p1])
    elif choice == 1:
        res[:] = np.dot(a[:,p0:p1], b[p0:p1], out=res)
    else:
        res[:,p0:p1] = np.dot(a, b[:,p0:p1], out=res[:,p0:p1])

    if squeeze_left and squeeze_right:
        res = res.ravel()[0]
    elif squeeze_left:
        res = res.squeeze(axis=0)
    elif squeeze_right:
        res = res.squeeze(axis=1)

    barrier()
    allreduce_inplace(res)

    return res

def tensordot(a, b, axes=2):
    # numpy.tensordot with MPI support

    try:
        iter(axes)
    except Exception:
        axes_a, axes_b = list(range(-axes, 0)), list(range(0, axes))
    else:
        axes_a, axes_b = axes

    try:
        na, axes_a = len(axes_a), list(axes_a)
    except TypeError:
        na, axes_a = 1, [axes_a]

    try:
        nb, axes_b = len(axes_b), list(axes_b)
    except TypeError:
        nb, axes_b = 1, [axes_b]

    a, b = np.asarray(a), np.asarray(b)
    as_, bs = a.shape, b.shape
    nda, ndb = a.ndim, b.ndim
    equal = True
    
    if na != nb:
        equal = False
    else:
        for k in range(na):
            if as_[axes_a[k]] != bs[axes_b[k]]:
                equal = False
                break
            if axes_a[k] < 0:
                axes_a[k] += nda
            if axes_b[k] < 0:
                axes_b[k] += ndb
    if not equal:
        raise ValueError('shape-mismatch for sum')

    notin = [k for k in range(nda) if k not in axes_a]
    newaxes_a = notin + axes_a
    N2 = np.prod([as_[axis] for axis in axes_a])
    newshape_a = (int(np.multiply.reduce([as_[ax] for ax in notin])), N2)
    olda = [as_[axis] for axis in notin]

    notin = [k for k in range(ndb) if k not in axes_b]
    newaxes_b = axes_b + notin
    N2 = np.prod([bs[axis] for axis in axes_b])
    newshape_b = (N2, int(np.multiply.reduce([bs[ax] for ax in notin])))
    oldb = [bs[axis] for axis in notin]

    at = a.transpose(newaxes_a).reshape(newshape_a)
    bt = b.transpose(newaxes_b).reshape(newshape_b)
    res = dot(at, bt)

    return res.reshape(olda + oldb)

def einsum(key, a, b):
    # single numpy.einsum contraction via tensordot with MPI support
    # the same axes must be contracted in a and b (order may vary)

    akey, bkey = key.split('->')[0].split(',')
    rkey = key.split('->')[1]
    rkey_default = ''

    axes_a = []
    axes_b = []

    for i,k in enumerate(akey):
        if k in rkey: 
            if k not in rkey_default:
                rkey_default += k
        else:
            axes_a.append(i)

    for i,k in enumerate(bkey):
        if k in rkey: 
            if k not in rkey_default:
                rkey_default += k
        else:
            axes_b.append(i)

    if len(axes_a) != len(axes_b):
        raise NotImplementedError

    keys_to_sum_a = ''.join([akey[x] for x in axes_a])
    keys_to_sum_b = ''.join([bkey[x] for x in axes_b])
    if keys_to_sum_a != keys_to_sum_b:
        axes_b = [axes_b[keys_to_sum_a.index(k)] for k in keys_to_sum_b]

    res = tensordot(a, b, axes=(tuple(axes_a[::-1]), tuple(axes_b[::-1])))

    if rkey != rkey_default:
        perm = [rkey_default.index(k) for k in rkey]
        res = res.transpose(*perm)

    return res


if __name__ == '__main__':
    keys = [
        'ibjc,jia->cab',
        'cab,icab->i',
        'ibjc,jia->cab',
        'cab,icab->i',
        'i,ibac->cab',
        'cab,jcib->ija',
    ]
    sizes = dict(i=5, j=5, k=5, l=5, a=5, b=5, c=5, d=5)

    for key in keys:
        a = np.random.random([sizes[x] for x in key.split(',')[0]])
        b = np.random.random([sizes[x] for x in key.split(',')[1].split('->')[0]])
        resa = np.einsum(key, a, b)
        resb =    einsum(key, a, b)
        print('%16s %8s' % (key, np.allclose(resa, resb)))
