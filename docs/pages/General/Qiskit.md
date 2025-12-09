---
title: "Qiskit/en"
url: "https://docs.alliancecan.ca/wiki/Qiskit/en"
category: "General"
last_modified: "2025-03-31T21:16:04Z"
page_id: 26302
display_title: "Qiskit"
---

`<languages />`{=html}

Developed in Python by IBM, [Qiskit](https://docs.quantum.ibm.com/) is an open-source quantum computing library. Like [PennyLane](https://docs.alliancecan.ca/PennyLane/en "PennyLane"){.wikilink} and [Snowflurry](https://docs.alliancecan.ca/Snowflurry/en "Snowflurry"){.wikilink}, it allows you to build, simulate and run quantum circuits.

## Installation

1\. Load the Qiskit dependencies.

2\. Create and activate a [ Python virtual environment](https://docs.alliancecan.ca/Python#Creating_and_using_a_virtual_environment " Python virtual environment"){.wikilink}.

3\. Install a version of Qiskit. X.Y.Z qiskit_aerX.Y.Z}} where `X.Y.Z` is the version number, for example `1.4.0`. To install the most recent version available on our clusters, do not specify a number. Here, we only imported `qiskit` and `qiskit_aer`. You can add other Qiskit software with the syntax `qiskit_package==X.Y.Z` where `qiskit_package` is the softare name, for example `qiskit-finance`. To see the wheels that are currently available, see [Available Python wheels](https://docs.alliancecan.ca/Available_Python_wheels "Available Python wheels"){.wikilink}.

4\. Validate the installation.

5\. Freeze the environment and its dependencies.

## Running Qiskit on a cluster {#running_qiskit_on_a_cluster}

```{=mediawiki}
{{File
  |name=script.sh
  |lang="sh"
  |contents=
#!/bin/bash
#SBATCH --account=def-someuser #Modify with your account name
#SBATCH --time=00:15:00        #Modify as needed
#SBATCH --cpus-per-task=1      #Modify as needed
#SBATCH --mem-per-cpu=1G       #Modify as needed

# Load module dependencies.
module load StdEnv/2023 gcc python/3.11 symengine/0.11.2 

# Generate your virtual environment in $SLURM_TMPDIR.                                                                                                         
virtualenv --no-download ${SLURM_TMPDIR}/env                                                                                                                   
source ${SLURM_TMPDIR}/env/bin/activate  

# Install Qiskit and its dependencies.                                                                                                                                                                                                                                                                                    
pip install --no-index --upgrade pip                                                                                                                            
pip install --no-index --requirement ~/qiskit_requirements.txt

# Modify your Qiskit program.                                                                                                                                                                       
python qiskit_example.py
}}
```
You can then [submit your job to the scheduler](https://docs.alliancecan.ca/Running_jobs "submit your job to the scheduler"){.wikilink}.

## Using Qiskit with MonarQ (in preparation) {#using_qiskit_with_monarq_in_preparation}

## Use case: Bell states {#use_case_bell_states}

Before you create a simulation of the first Bell state on [Narval](https://docs.alliancecan.ca/Narval/en "Narval"){.wikilink}, the required modules need to be loaded.

`   from qiskit_aer import AerSimulator`\
`   from qiskit import QuantumCircuit, transpile`\
`   from qiskit.visualization import plot_histogram`

Define the circuit. Apply an Hadamard gate to create a superposition state on the first qubit and a CNOT gate to intricate the first and second qubits.

`   circuit = QuantumCircuit(2,2)`\
`   circuit.h(0)`\
`   circuit.cx(0,1)`\
`   circuit.measure_all()`

Nous voulons utiliser le simulateur par défaut, soit `AerSimulator` étant le simulateur par défaut. Nous obtenons le dénombrement des états finaux des qubits après 1000 mesures.

`   simulator = AerSimulator()`\
`   result = simulator.run(circuit, shots=1000).result()`\
`   counts = result.get_counts()`\
`   print(counts)`\
`   {'00': 489, '11': 535}`

Nous affichons un histogramme des résultats avec la commande

`   plot_histogram(counts)`

We will use the default simulator `AerSimulator`. This provides the final number of qubits after having made 1000 measurements.

`   simulator = AerSimulator()`\
`   result = simulator.run(circuit, shots=1000).result()`\
`   counts = result.get_counts()`\
`   print(counts)`\
`   {'00': 489, '11': 535}`

The results are displayed.

`   plot_histogram(counts)`

\[\[<File:Qiskit> counts.png\|thumb\|Results of 1000 measurements on the first Bell

`state]]`
