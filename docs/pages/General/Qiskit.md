---
title: "Qiskit/en"
url: "https://docs.alliancecan.ca/wiki/Qiskit/en"
category: "General"
last_modified: "2025-03-31T21:16:04Z"
page_id: 26302
display_title: "Qiskit"
---

Developed in Python by IBM, Qiskit is an open-source quantum computing library. Like PennyLane and Snowflurry, it allows you to build, simulate and run quantum circuits.

== Installation ==
1. Load the Qiskit dependencies.

2. Create and activate a  Python virtual environment.

3. Install a version of Qiskit.
X.Y.Z  qiskit_aerX.Y.Z}}
where X.Y.Z is the version number, for  example 1.4.0. To install the most recent version available on our clusters, do not specify a number. Here, we only imported qiskit and qiskit_aer. You can add other Qiskit software with the syntax qiskit_package==X.Y.Z where qiskit_package is the softare name, for example qiskit-finance. To see the wheels that are currently available, see Available Python wheels.

4. Validate the installation.

5. Freeze the environment and its dependencies.

==Running Qiskit on a cluster==

You can then submit your job to the scheduler.
== Using Qiskit with MonarQ (in preparation)==

== Use case: Bell states ==
Before you create a simulation of the first Bell state on Narval, the required modules need to be loaded.
    from qiskit_aer import AerSimulator
    from qiskit import QuantumCircuit, transpile
    from qiskit.visualization import plot_histogram

Define the circuit. Apply an Hadamard gate to create a superposition state on the first qubit and a CNOT gate to intricate the first and second qubits.
    circuit = QuantumCircuit(2,2)
    circuit.h(0)
    circuit.cx(0,1)
    circuit.measure_all()

Nous voulons utiliser le simulateur par défaut, soit AerSimulator étant le simulateur par défaut. Nous obtenons le dénombrement des états finaux des qubits après 1000 mesures.
    simulator = AerSimulator()
    result = simulator.run(circuit, shots=1000).result()
    counts = result.get_counts()
    print(counts)
    {'00': 489, '11': 535}
Nous affichons un histogramme des résultats avec la commande
    plot_histogram(counts)

We will use the default simulator AerSimulator. This provides the final number of qubits after having made 1000 measurements.
    simulator = AerSimulator()
    result = simulator.run(circuit, shots=1000).result()
    counts = result.get_counts()
    print(counts)
    {'00': 489, '11': 535}
The results are displayed.
    plot_histogram(counts)