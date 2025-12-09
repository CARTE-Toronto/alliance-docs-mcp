---
title: "Snowflurry/en"
url: "https://docs.alliancecan.ca/wiki/Snowflurry/en"
category: "General"
last_modified: "2024-10-24T20:38:45Z"
page_id: 26343
display_title: "Snowflurry"
---

Snowflurry is an open-source quantum computing library developed in  Julia by Anyon Systems that allows you to build, simulate, and run quantum circuits. A related library called SnowflurryPlots allows you to visualize the simulation results in a bar chart. Useful to explore quantum computing, its features are described in the documentation and the installation guide is available on the GitHub page. Like the PennyLane library, Snowflurry can be used to run quantum circuits on the MonarQ quantum computer.

== Installation ==
The quantum computer simulator with Snowflurry is available on all of our clusters. The Julia programming language  must be loaded before accessing Snowflurry.
    } }}|lang=}}}

The Julia programming interface is then called and the Snowflurry quantum library is loaded (in about 5-10 minutes) with the commands
    } }}|lang=}}}

Quantum logic gates and commands are described in the Snowflurry documentation.

== Use case: Bell states ==
Bell states are maximally entangled two-qubit states. They are simple examples of two quantum phenomena: superposition and entanglement. The Snowflurry library allows you to construct the first Bell state as follows:

In the above code section, the Hadamard gate creates an equal superposition of |0⟩ and |1⟩ on the first qubit while the CNOT gate (controlled X gate) creates an entanglement between the two qubits. We find an equal superposition of states |00⟩ and |11⟩, which is the first Bell state. The simulate function allows us to simulate the exact state of the system.

  julia> state = simulate(circuit)
  julia> print(state)
  4-element Ket{ComplexF64}:
  0.7071067811865475 + 0.0im
  0.0 + 0.0im
  0.0 + 0.0im
  0.7071067811865475 + 0.0im

The readout operation lets you specify which qubits will be measured. The plot_histogram function from the SnowflurryPlots library allows you to visualize the results.