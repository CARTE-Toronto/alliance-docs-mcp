---
title: "CirQ/en"
url: "https://docs.alliancecan.ca/wiki/CirQ/en"
category: "General"
last_modified: "2024-10-16T15:55:59Z"
page_id: 26685
display_title: "CirQ"
---

Developed by Google, CirQ  is an open-source quantum computing library to build, optimize, simulate and run quantum circuits. More specifically, CirQ allows to simulate circuits on particular qubit configurations, which can optimize a circuit for a certain qubit architecture. Information on the features can be found in the CirQ documentation and GitHub. Like  Snowflurry, CirQ can be used to run quantum circuits on the MonarQ quantum computer.

== Installation ==
The CirQ simulator is available on all of our clusters. To have access, you must load the Python language. Il est préférable de travailler dans un environnement virtuel Python.
1.4.1
|python -c "import cirq"
|pip freeze > cirq-1.4.1-reqs.txt
}}
The last command creates the cirq-1.4.1-reqs.txt file which you can also use in a job script such as in the example below.
==Exécution sur une grappe==

You can then submit your job to the scheduler.

== Use case: Bell states ==
Les états de Bell sont les états les plus simples qui permettent d'expliquer à la fois la superposition et l'intrication sur des qubits.
La bibliothèque CirQ permet de construire un état de Bell comme ceci :

This code builds and displays a circuit that prepares a Bell state. The H gate (Hadamard gate) creates an equal superposition of |0⟩ and |1⟩ on the first qubit while the CNOT gate (controlled X gate) creates an entanglement between the two qubits. This Bell state is therefore an equal superposition of the states |00⟩ and |11⟩. Simulating this circuit using CirQ allows you to visualize the results. In this diagram, the integer 3 represents the state |11⟩ since 3 is written 11 in binary.