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

def dot(a, b):
    # numpy.dot with MPI support

    if a.ndim == 1:
        a = a.reshape(1, -1)
    elif a.ndim > 2:
        raise ValueError

    if b.ndim == 1:
        b = b.reshape(-1, 1)
    elif b.ndim > 2:
        raise ValueError

    assert a.shape[1] == b.shape[0]
    axes = (a.shape[0], a.shape[1], b.shape[0])
    choice = np.argmax(axes)
    p0, p1 = distr_blocks(axes[choice])
    res = np.zeros((a.shape[0], b.shape[1]))

    if choice == 0:
        res[p0:p1] = np.dot(a[p0:p1], b)
    elif choice == 1:
        res = np.dot(a[:,p0:p1], b[p0:p1])
    else:
        res[:,p0:p1] = np.dot(a, b[:,p0:p1])

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

    res = tensordot(a, b, axes=(tuple(axes_a[::-1]), tuple(axes_b[::-1])))

    if rkey != rkey_default:
        perm = [rkey_default.index(k) for k in rkey]
        res = res.transpose(*perm)

    return res

