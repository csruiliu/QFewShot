# Fidelity-Based Quantum-Classical Few-Shot Learning

<p align = center>
<img src="Assets/circuit icon.png" width="400">

This project was submitted for the iQuHACK 2022 challenge. 

Humans learn new concepts with very little supervision – e.g. a child can generalize the concept of “giraffe” from a single picture in a book – yet our best deep learning systems need hundreds or thousands of examples. [^1] Few-shot classification is a task in which a classifier must be adapted to accommodate new classes not seen in training, given only a few examples of each of these classes. [2] In practice, few-shot learning is useful when training examples are hard to find (e.g., cases of a rare disease), or where the cost of labeling data is high.

### Table of Contents  
1. [Introduction to Machine Learning and Quantum Machine Learning](#toc1)
2. [Building the Machine Learning Pipeline](#toc2)
3. [QTensor/QTensorAI and integration with the Few-Shot Model](#toc3)
4. [Quantum Circuit for the calculation of Inner Product](#toc4)
5. [Medical Dataset and Real World Application](#toc5)
6. [Results](#toc6)
7. [Installation Guide](#toc7)
8. [References](#toc8)

<a name="toc1"></a>

## Introduction to Machine Learning and Quantum Machine Learning

State-of-the-art models utilize sampled mini-batches called episodes during training, where each episode is designed to mimic the few-shot task by subsampling classes as well as data points. [^2] The model, therefore, learns to extract embeddings for each sample that are useful for different few-shot tasks. The embeddings of the examples and the queries are compared by a distance metric, and the predicted class minimizes such distances.

We replace the classical model with a hybrid quantum-classical embedding. Images are classically processed through convolution, and the final low-dimensional vectors can be used as variational parameters to a quantum circuit. The model essentially learns embeddings in the quantum Hilbert space. The high dimensionality of the Hilbert space of the circuit means that the circuit can potentially model kernels that are classically hard to model. This is the quantum kernel method [^3]. The distance between the quantum embeddings can be then estimated as the square of the inner product of their respective wavefunctions.

We simulate large quantum circuits with QTensor [^4], a tensor network simulator [^5]. Tensor networks move the simulation complexity to (in a sense) the depth of the circuit and are linear in the number of qubits. There is great potential in demonstrating the feasibility of this approach using this simulator.

Additionally, we are going to run our hybrid quantum-classical model on the provided quantum hardware to see the performance differences between the simulation and the physical machine. Our aim is to run the model on the backends with different modalities: Trapped Ion from IonQ and Superconducting qubits from IBM. The purpose of this comparison is to benchmark hardware against a simulation to see the effects of noise and corresponding error correction and to build on the 2017 comparison between these quantum architectures[^6]. 

Our project will be building on top of [the code used for the "Prototypical networks for few-shot learning." by Snell et al.](https://github.com/jakesnell/prototypical-networks)

| | |
| :--------------: | :---------: 
| <img src="Assets/img1.png" width="400"> | <img src="Assets/img2.png" width="400">

<a name="toc2"></a>
## Building the Machine Learning Pipeline

<img src="Assets/mlgif1.gif" width="800">

<a name="toc3"></a>
## QTensor/QTensorAI and integration with the Few-Shot Model

Quantum computers are hard to simulate classically, and this is a good thing in the long run,
since we wouldn't need quantum computing otherwise. However we do need to simulate quantum computers
classically to design and fine-tune quantum algorithms. This is where tensor networks [^8] come to help.
 
This is an example tensor network:

<img src="Assets/tensornet.png" width="400">

The simulation of quantum circuit is limited by memory required to simulate. Tensor networks allow to 
take advantage of structure of a quantum circuit to reduce memory footprint of the simulation[^9].

In our code, we use a tensor network-based quantum simulator "QTensor". It is capable of simulating much larger circuits than state vector simulators. For example our simulation of 50 qubits would require over 4000 Terabytes of memory with statevector. Quantum operations are treated as tensors whose indices corresponds to their input and output qubits.

This picture shows how quantum gates are represented as a undirected graph, which is then contracted optimally using tensor netwokr contraction algorithms:

| Diagonal Gates | Non-Diagonal Gates |
| :--------------: | :---------: 
| <img src="Assets/diagonal_gates.png" width="400"> | <img src="Assets/non_diagonal_gates.png" width="400">|

Note that if the gate is diagonal in the computational basis, it is possible to simplify the calculations.

Please view [this code](./QTensorAI) for the technical implementation of this section. 

QTensorAI is a library that enables the utilization of the QTensor quantum simulator within PyTorch machine learning pipelines. The library offers a few circuit ansatz simulators wrapped as `nn.Module` classes, and other tools to allow users to construct their own ML-friendly circuit ansatzes. Further, QTensorAI changes the dynamic implementation of QTensor to static, making it friendly to CUDA Graph optimization, which eliminates CPU side kernel launch overhead and can lead to a significant speed-up. Combined with the added capability of batch parallelism, QTensorAI optimizes large machine learning tasks and removes significant overhead in quantum simulations.

<img src="Assets/qimg1.png" width="400">

| 50 qubit 8 variational layer prototypical networks | 32 qubit 4 variational layer temporal convolution |
| :--------------: | :---------: 
| <img src="Assets/50qubit8varproto.png" width="400"> | <img src="Assets/32qubit4vartemp.png" width="400">|


<a name="toc4"></a>
## Quantum Circuit for the calculation of Inner Product
A circuit inspired by Linear Entanglement Ansatz [^10] with alternating layers of single qubit rotation and CNOT gates is employed to generate an Unitary Operator corresponding to a feature vector. The elements of the feature vector are encoded as rotation angles for Y and Z rotation gates. The Unitary generated from a given feature vector maps the all zero state to a state in the Hilbert space. 

In order to decide whether a query represented by a feature vector belongs to one of the support classes, we compute the square of the inner product for the states corresponding to the query with those learned from the support class. Fortunately, at the cost of doubling the depth of the circuit this can be done by composing the circuit for generating the unitary for the query with the adjoint of the circuit generating the unitary for the support class. Finally all the qubits are measured and statistics for the overlap are collected.  The following figure illustrates this idea for 10 qubits with a feature vector of length 40.



<p align = center>
<img src="Assets/qiskit_circuit.png" width="800">

 
Please view [this notebook](./azure_ionq/Running_Circuits.ipynb) for the technical implementation of this section. 
 
```py
# trial to build ansatz and do inner product 
# the rotation array has Rz,Ry alternating for each qubit
# arranged qubit by qubit so it as an array of arrays 
# with inner array length of 2*repetitions and outer array of length n_qubits

def qfl_constructor(rot_array):
    n_qubits = len(rot_array)
    repetitions = int(len(rot_array[0])/2)
    circuit = QuantumCircuit(n_qubits,n_qubits)
    # we will have two angles per repetition per qubit
    for j in np.arange(0,2*repetitions,2):
        for i in range(n_qubits):
            circuit.rz(phi=rot_array[i][j], qubit=i)    
    
        for i in range(n_qubits):
            circuit.ry(theta=rot_array[i][j+1], qubit=i)

        circuit.barrier()
    
        for i in np.arange(1,n_qubits,2):
            circuit.cx(control_qubit=i-1, target_qubit=i)
    
        for i in np.arange(2,n_qubits,2):
            circuit.cx(control_qubit=i-1, target_qubit=i)
    

    return circuit

# this function spits out the combined circuit based on arrays specifying the rotation gates


def qfl_combined_constructor(rot_array_first, rot_array_second):
    circuit_first = qfl_constructor(rot_array_first)
    circuit_second = qfl_constructor(rot_array_second)
    inverse_circuit_second = circuit_second.inverse()
    circuit_combined = circuit_first.compose(inverse_circuit_second)
    # measure all the qubits at the end to get probability of getting 0000...00 
    circuit_combined.measure_all()
    return circuit_combined
```

<a name="toc5"></a>
## Medical Dataset and Real World Application

<a name="toc6"></a>
## Results
Omniglot is a dataset of 1623 handwritten characters collected from 50 alphabets. There are 20 examples associated with each character, where each example is drawn by a different human subject. Our 60-way 5-shot model is trained testing accuracy is  sdfafsaf  percent, where random guessing gives 1/60*100%=1.7% accuracy. 
The miniImageNet dataset consists of 60,000 color images of size 84 × 84 divided into 100 classes with 600 examples each. All classes are divided into 64 training, 16 validation, and 20 test classes. Our 5-way 5-shot model is trained with a 50-qubit 8 variational-layer circuit, resulting in a validation accuracy of 48.71%. Random guessing is expected to give a ⅕*100% = 20% accuracy.
| | |
| :--------------: | :---------: 
| <img src="Assets/_         Validation Accuracy for Omniglot 60-way 5-shot.png" width="400"> | <img src="Assets/_         Validation Accuracy for MiniImageNet 5-way 5-shot.png" width="400">|
 
<a name="toc7"></a>

## Installation Guide ##

### QTensor Installation ###

First, you need to install QTensor from [source code](https://github.com/danlkv/qtensor).

```bash
# --recurse-submodules is important since qtensor has a submodule qtree  
git clone --recurse-submodules https://github.com/DaniloZZZ/QTensor
cd QTensor

# using git branch -v -a to check all branchs
# and switch to dev branch
git switch dev
# checkout the specific commit 
git checkout dc53509

cd qtree
pip install .

cd .. # back to QTensor folder
pip install .
```

You can check if the `qtensor` and `qtree` are installed or not using `pip3 list`

### QTensorAI Installation ###

```bash
cd QTensorAI
python3 setup.py install
```

You can check if the `qtensor-ai` is installed or not using `pip3 list`

<a name="toc8"></a>

## References
[^1]: Vinyals, Oriol, Charles Blundell, Timothy Lillicrap, and Daan Wierstra. "Matching networks for one shot learning." Advances in neural information processing systems 29 (2016): 3630-3638.
[^2]: Snell, Jake, Kevin Swersky, and Richard S. Zemel. "Prototypical networks for few-shot learning." arXiv preprint arXiv:1703.05175 (2017).
[^3]: Havlíček, Vojtěch, Antonio D. Córcoles, Kristan Temme, Aram W. Harrow, Abhinav Kandala, Jerry M. Chow, and Jay M. Gambetta. "Supervised learning with quantum-enhanced feature spaces." Nature 567, no. 7747 (2019): 209-212.
[^4]: Lykov, Danylo, Roman Schutski, Alexey Galda, Valerii Vinokur, and Yurii Alexeev. "Tensor network quantum simulator with step-dependent parallelization." arXiv preprint arXiv:2012.02430 (2020).
[^5]: Boixo, Sergio, Sergei V. Isakov, Vadim N. Smelyanskiy, and Hartmut Neven. "Simulation of low-depth quantum circuits as complex undirected graphical models." arXiv preprint arXiv:1712.05384 (2017).
[^6]: Linke, Norbert M., Dmitri Maslov, Martin Roetteler, Shantanu Debnath, Caroline Figgatt, Kevin A. Landsman, Kenneth Wright, and Christopher Monroe. "Experimental comparison of two quantum computing architectures." Proceedings of the National Academy of Sciences 114, no. 13 (2017): 3305-3310.
[^7]: Koch, Gregory R.. “Siamese Neural Networks for One-Shot Image Recognition.” (2015).
[^8]: [Jacob Biamonte and Ville Bergholm "Tensor Networks in a Nutshell"](https://arxiv.org/abs/1708.00006)
[^9]: "Tensor Network Quantum Simulator With Step-Dependent Parallelization
" https://arxiv.org/abs/2012.02430
 [^10]: https://qiskit.org/textbook/ch-applications/vqe-molecules.html#simplevarform
