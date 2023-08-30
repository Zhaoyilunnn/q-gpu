## !! This repo will be deprecated !!
We are implementing Q-GPU in a seperate package which is decoupled with qiskit-aer, and will be more portable, stay tuned on [qdao](https://github.com/Zhaoyilunnn/qdao)

## Build from source

To build this project, please follow the instructions in the [contribution guidelines](https://github.com/Qiskit/qiskit-aer/blob/master/CONTRIBUTING.md). After setting up prerequisite env, run the following commands for compilation.

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
| compression | w/ compression (num\_qubits >= 21) |

## References

Q-GPU is developped based on [QISKit-Aer](https://github.com/Qiskit/qiskit-aer), [0.7.0](https://github.com/Qiskit/qiskit-aer/tree/0.7.0). The original version is the baseline of paper.

We use [GFC](https://userweb.cs.txstate.edu/~burtscher/research/GFC/) as data compression algorithm in Q-GPU. (With small modifications to adapt to Q-GPU.)
