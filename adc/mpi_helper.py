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
