---
title: "PennyLane/en"
url: "https://docs.alliancecan.ca/wiki/PennyLane/en"
category: "General"
last_modified: "2025-07-02T14:37:29Z"
page_id: 26337
display_title: "PennyLane"
---

PennyLane is an open-source software platform for differentiable quantum computing that was launched in 2018 by Xanadu, a quantum technology company based in Toronto, Canada. It allows quantum circuits to be designed and run on a variety of quantum simulators and hardware. PennyLane is designed to facilitate the simulation, optimization, and training of hybrid quantum algorithms, which combine classical and quantum processing. The first version was published as an open-source project on GitHub.

==Features==
PennyLane offers several features to facilitate research and development in differentiable quantum computing.

==Unified quantum interface==
PennyLane provides a unified quantum interface that allows you to design quantum circuits and run them on different quantum simulators and hardware. PennyLane supports several popular quantum simulators, such as Qiskit, CirQ, Strawberry Field, and QuTip. PennyLane also supports several quantum hardware, including Xanadu, IBM, Rigetti and IonQ quantum devices.

Calcul Québec has developed a PennyLane-CalculQuebec plugin that uses the PennyLane interface to design and run quantum circuits on MonarQ.

===Integration with machine learning libraries===
PennyLane seamlessly integrates with popular machine learning libraries such as TensorFlow and PyTorch, allowing you to use machine learning tools to build hybrid quantum machine learning models and optimize quantum circuits.

===Quantum circuit optimization===
Using differentiable optimization techniques and combining classical and quantum differentiation methods, PennyLane optimizes quantum circuit parameters to solve a variety of problems

=== Visualization tools ===
PennyLane provides visualization tools to help understand how quantum circuits work.

=== Community and development ===
PennyLane is an open-source project with an active community of developers and users. The project is constantly updated with new features and improvements, and anyone can contribute to the development of the platform.

== Using PennyLane with MonarQ ==
MonarQ is designed to be programmed with Snowflurry, a Julia-based software library developed by Anyon Systems. However, with the PennyLane-CalculQuebec plugin, PennyLane circuits can be created using Snowflurry in the background. This allows circuits to be run on MonarQ while still benefiting from the features and development environment offered by PennyLane. See the PennyLane-CalculQuebec documentation for installation and usage guides.

A quantum transpiler is also available to optimize PennyLane circuits on MonarQ.

== Creating a virtual environment ==
Let’s create a virtual environment to use PennyLane.
X.Y.Z
|python -c "import pennylane"
}}
where X.Y.Z is the version number.

You can also put the last three commands in a pennylane-reqs.txt file and call the file inside a session with the commands

==Running PennyLane on a cluster==

You can now submit the job to the scheduler.

== Use case: Bell states ==
Let's start by creating the virtual environment, as described above.

We will then generate the first Bell state using PennyLane.
    import pennylane as qml

   #  Define the quantum circuit to generate the first Bell state.
    def bell_circuit():
     qml.Hadamard(wires=0)
     qml.CNOT(wires=[0, 1])

   # Define the quantum circuit simulator.
    dev = qml.device('default.qubit', wires=2)

   # Define the quantum circuit as a QNode function.
    @qml.qnode(dev)
    def generate_bell_state():
     bell_circuit()
     return qml.state()

   # Generate and display the first Bell state.
    bell_state_0 = generate_bell_state()
   print("First Bell State :", bell_state_0)
   Premier état de Bell :[0.70710678+0.j 0.        +0.j 0.        +0.j 0.70710678+0.j]

== References ==
* PennyLane official website
* PennyLane documentation on GitHub
* PennyLane-CalculQuebec