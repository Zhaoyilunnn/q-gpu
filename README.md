## Build from source

Please follow the instructions in the [contribution guidelines](https://github.com/Qiskit/qiskit-aer/blob/master/CONTRIBUTING.md).

```bash
python ./setup.py bdist_wheel -- -DAER_THRUST_BACKEND=CUDA -DCMAKE_CUDA_COMPILER=${YOUR_NVIDIA_COMPILER_PATH}
```

To run experiments, please see [this repo](https://github.com/Zhaoyilunnn/q-gpu-exp)

To enable multi-GPU simulation, first set up this environment variable

```bash
export AER_MULTI_GPU=1
```

## Branches 

| Branch name | Description |
| --- | --- |
| master | w/o compression |
| compression | w/ compressin |

## References

Q-GPU is developped by revising [QISKit-Aer](https://github.com/Qiskit/qiskit-aer). The baseline in the paper is [0.7.0](https://github.com/Qiskit/qiskit-aer/tree/0.7.0).

We use [GFC](https://userweb.cs.txstate.edu/~burtscher/research/GFC/) as data compression algorithm in Q-GPU.


