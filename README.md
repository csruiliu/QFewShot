# Fidelity-Based Quantum-Classical Few-Shot Learning

<p align = center>
<img src="Assets/circuit icon.png" width="400">

This project was submitted for the iQuHACK 2022 challenge. 

Humans learn new concepts with very little supervision – e.g. a child can generalize the concept of “giraffe” from a single picture in a book – yet our best deep learning systems need hundreds or thousands of examples. [^1] Few-shot classification is a task in which a classifier must be adapted to accommodate new classes not seen in training, given only a few examples of each of these classes. [^2] In practice, few-shot learning is useful when training examples are hard to find (e.g., cases of a rare disease), or where the cost of labeling data is high.

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

We have constructed an end-to-end machine learning pipeline to train the few-shot models with various datasets and deploy them for serving real-world applications. The pipeline consists of four main stages: *Pre-processing, Training, Evaluation, and Prediction*. Pre-processing Real-world data is usually incomplete, inconsistent, and frequently contains inaccuracies. The stage of data preprocessing could overcome this issue by cleaning the dataset and transferring raw data into an computable format for training. 
 
The *Pre-processing* consists of various data operations such as feature extraction, feature selection, dimensionality reduction, dimensionality rotation, scaling and sampling. The product of our data pre-processing is the final dataset used for training the model and testing purposes. The *Models are Trained* using different hyperparameter settings, metrics, and cross-validation techniques. For our *Model Evaluation*, the pipeline will pull the models from the training data and push them into evaluation and test dataset for prediction. Specifically, the pipeline counts the number of wrong predictions on the evaluation test dataset to compute the model’s prediction accuracy.   Finally, we are going to deploy the *Model for Prediction*. The pipeline will store the best-performance model with its meta like model architecture and parameters. When the model is deployed in the real-world application, the pipeline will load the model and initialize all the parameters for fine-tuning and prediction.
 
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


| 50 qubit 8 variational layer prototypical networks | 32 qubit 4 variational layer temporal convolution |
| :--------------: | :---------: 
| <img src="Assets/50qubit8varproto.png" width="200"> | <img src="Assets/32qubit4vartemp.png" width="200">|


<a name="toc4"></a>
## Quantum Circuit for the calculation of Inner Product
A circuit inspired by Linear Entanglement Ansatz [^10] with alternating layers of single qubit rotation and CNOT gates is employed to generate an Unitary Operator corresponding to a feature vector. The elements of the feature vector are encoded as rotation angles for Y and Z rotation gates. The Unitary generated from a given feature vector maps the all zero state to a state in the Hilbert space. The unitaries corresponding to the support classes are learned via training on a classical computer, making this a hybrid classical-quantum approach. 

In order to decide whether a query represented by a feature vector belongs to one of the support classes, we compute the square of the inner product for the states corresponding to the query with those learned from the support class. Fortunately, at the cost of doubling the depth of the circuit this can be done by composing the circuit for generating the unitary for the query with the adjoint of the circuit generating the unitary for the support class. Finally all the qubits are measured and statistics for the overlap are collected.  The following figure illustrates this idea for 10 qubits with a feature vector of length 40.



<p align = center>
<img src="Assets/qiskit_circuit.png" width="800">

 
Please view [this notebook](./azure_ionq/Running_Circuits.ipynb) for the technical implementation of this section. 

<a name="toc5"></a>
## Medical Dataset and Real World Application

Our project allows to extend the few-shot learning approach with the richness of quantum computing Hilbert space. One of the great applications of the few-shot learning is medical data, since the health data is very sensitive and hard to obtain on scale.

We apply our algorithm to a dataset of upper-body thermal images correlated with COVID-19 symptoms and PCR test results[^11].
The dataset contains infrared-camera videos of 252 people as well as description of each patient.

This is an example picture from the dataset:

<img src="Assets/heatimage.png" width="400">

(The sun sticker is not included in the dataset)

The description included various information including:

| Data section                       | Fields                                                                                                                                 |
|------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| Participant information            | ID, Age, Gender, Weight, Height, Last menstruation (LMP)                                                                               |
| SARS-CoV-2 Exposure                | House, Hospital/Clinics, School/Work, Other                                                                                            |
| Vital signs                        | Temperature, Blood pressure, Cardiac rate, O2 Saturation, Respiratory rate                                                             |
| Symptoms (< 24 hours)              | Fever, Cough, Throat pain, Diarrhea, Vomit, Smell loss, Taste loss, Shivers/chills, Headache, Myalgia, Generalized Arthralgias, Others |
| PCR Diagnosis*                     | Result, (Estimated viral load)                                                                                                         |

We use this dataset to predict a set of symptoms by the thermal image. The source code for it is located [here](few-shot/covid_prediction_ext)

<a name="toc6"></a>
## Results
Omniglot is a dataset of 1623 handwritten characters collected from 50 alphabets. There are 20 examples associated with each character, where each example is drawn by a different human subject. Our 60-way 5-shot model is trained testing accuracy is 98% percent, where random guessing gives 1/60*100%=1.7% accuracy. 
The miniImageNet dataset consists of 60,000 color images of size 84 × 84 divided into 100 classes with 600 examples each. All classes are divided into 64 training, 16 validation, and 20 test classes. Our 5-way 5-shot model is trained with a 50-qubit 8 variational-layer circuit, resulting in a validation accuracy of 48.71%. Random guessing is expected to give a ⅕*100% = 20% accuracy.
| | |
| :--------------: | :---------: 
| <img src="Assets/_         Validation Accuracy for Omniglot 60-way 5-shot.png" width="400"> | <img src="Assets/_         Validation Accuracy for MiniImageNet 5-way 5-shot.png" width="400">|
 
<a name="toc7"></a>

## Installation Guide ##

For installation of [QTensorAI](./QTensorAI/README.md) and [few-shot](./few-shot/README.md), see the respective folders for their readme.

<a name="toc8"></a>
 
 
 ### Personal Experience ###

We had a lot of fun working on Machine Learning, Microsoft Azure and running circuits on the Ion Q computer. This was a wonderful learning oppurtunity and we thank the organizers for giving us the oppurtunity to participate in the Hackathon. 

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
[^11]: [Upper body thermal images and associated clinical data from a pilot cohort study of COVID-19](https://physionet.org/content/covid-19-thermal/1.1/)
