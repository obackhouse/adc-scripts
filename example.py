from pyscf import gto, scf
import adc
import time

mol = gto.M(atom='O 0 0 0; O 0 0 1', basis='cc-pvtz', verbose=0)
rhf = scf.RHF(mol).run(conv_tol=1e-12)

t0 = time.time()
e_ip = adc.run(rhf, method='2', which='ip')[0]
e_ea = adc.run(rhf, method='2', which='ea')[0]
t1 = time.time()
if adc.mpi_helper.rank == 0:
    print('Exact ERIs:')
    print('IP:', e_ip)
    print('EA:', e_ea)
    print('Time:', t1-t0)

rhf = rhf.density_fit(auxbasis='aug-cc-pvqz-jkfit')
rhf.run(conv_tol=1e-12)

t0 = time.time()
e_ip = adc.run(rhf, method='2', which='ip')[0]
e_ea = adc.run(rhf, method='2', which='ea')[0]
t1 = time.time()
if adc.mpi_helper.rank == 0:
    print('Density fitting:')
    print('IP:', e_ip)
    print('EA:', e_ea)
    print('Time:', t1-t0)
