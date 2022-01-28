# Fidelity-Based Quantum-Classical Few-Shot Learning

Humans learn new concepts with very little supervision – e.g. a child can generalize the concept of “giraffe” from a single picture in a book – yet our best deep learning systems need hundreds or thousands of examples. [1] Few-shot classification is a task in which a classifier must be adapted to accommodate new classes not seen in training, given only a few examples of each of these classes. [2] In practice, few-shot learning is useful when training examples are hard to find (e.g., cases of a rare disease), or where the cost of labeling data is high.

State-of-the-art models utilize sampled mini-batches called episodes during training, where each episode is designed to mimic the few-shot task by subsampling classes as well as data points. [2] The model, therefore, learns to extract embeddings for each sample that are useful for different few-shot tasks. The embeddings of the examples and the queries are compared by a distance metric, and the predicted class minimizes such distances.

We replace the classical model with a hybrid quantum-classical embedding. Images are classically processed through convolution, and the final low-dimensional vectors can be used as variational parameters to a quantum circuit. The model essentially learns embeddings in the quantum Hilbert space. The high dimensionality of the Hilbert space of the circuit means that the circuit can potentially model kernels that are classically hard to model. This is the quantum kernel method [3]. The distance between the quantum embeddings can be then estimated as the square of the inner product of their respective wavefunctions.

We simulate large quantum circuits with QTensor [4], a tensor network simulator [5]. Tensor networks move the simulation complexity to (in a sense) the depth of the circuit and are linear in the number of qubits. There is great potential in demonstrating the feasibility of this approach using this simulator.

Additionally, we are going to run our hybrid quantum-classical model on the provided quantum hardware to see the performance differences between the simulation and the physical machine. Our aim is to run the model on the backends with different modalities: Trapped Ion from IonQ and Superconducting qubits from IBM. The purpose of this comparison is to benchmark hardware against a simulation to see the effects of noise and corresponding error correction and to build on the 2017 comparison between these quantum architectures[6]. 

Our project will be building on top of [the code used for the "Prototypical networks for few-shot learning." by Snell et al.](https://github.com/jakesnell/prototypical-networks)

<p align="left"><img src="Assets/img1.png"/width=500></p>
<p align="left"><img src="Assets/img2.png"/width=500></p>


## References
1. Vinyals, Oriol, Charles Blundell, Timothy Lillicrap, and Daan Wierstra. "Matching networks for one shot learning." Advances in neural information processing systems 29 (2016): 3630-3638.
2. Snell, Jake, Kevin Swersky, and Richard S. Zemel. "Prototypical networks for few-shot learning." arXiv preprint arXiv:1703.05175 (2017).
3. Havlíček, Vojtěch, Antonio D. Córcoles, Kristan Temme, Aram W. Harrow, Abhinav Kandala, Jerry M. Chow, and Jay M. Gambetta. "Supervised learning with quantum-enhanced feature spaces." Nature 567, no. 7747 (2019): 209-212.
4. Lykov, Danylo, Roman Schutski, Alexey Galda, Valerii Vinokur, and Yurii Alexeev. "Tensor network quantum simulator with step-dependent parallelization." arXiv preprint arXiv:2012.02430 (2020).
5. Boixo, Sergio, Sergei V. Isakov, Vadim N. Smelyanskiy, and Hartmut Neven. "Simulation of low-depth quantum circuits as complex undirected graphical models." arXiv preprint arXiv:1712.05384 (2017).
6. Linke, Norbert M., Dmitri Maslov, Martin Roetteler, Shantanu Debnath, Caroline Figgatt, Kevin A. Landsman, Kenneth Wright, and Christopher Monroe. "Experimental comparison of two quantum computing architectures." Proceedings of the National Academy of Sciences 114, no. 13 (2017): 3305-3310.
7. Koch, Gregory R.. “Siamese Neural Networks for One-Shot Image Recognition.” (2015).
