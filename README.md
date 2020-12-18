A collection of scripts for the Algebraic Diagrammatic Construction method, some with basic MPI parallelisation.
I do not intend these to be fully featured or competitive in speed, but they are reasonably efficient and very lightweight.
It's mostly just a collection of reference implementations and for use in my own work.
For more robust implementations use [adcc](https://github.com/adc-connect/adcc) or the `pyscf.adc` [module](https://github.com/pyscf/pyscf/tree/master/pyscf/adc).

Features (M = MPI, R = Restricted, U = Unrestricted):

| ADC(2) | IP | EA | EE |
|--------|:--:|:--:|:--:|
| ERI    | MR | MR |    |
| DF     | MR | MR |    |
| PBC    | MR | MR |    |

| ADC(2)-x | IP | EA | EE |
|----------|:--:|:--:|:--:|
| ERI      | MR | MR |    |
| DF       |    |    |    |
| PBC      |    |    |    |

| ADC(3) | IP | EA | EE |
|--------|:--:|:--:|:--:|
| ERI    | R  | R  |    |
| DF     |    |    |    |
| PBC    |    |    |    |
