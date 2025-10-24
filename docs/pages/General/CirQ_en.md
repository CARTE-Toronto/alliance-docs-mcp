---
title: "CirQ/en"
url: "https://docs.alliancecan.ca/wiki/CirQ/en"
category: "General"
last_modified: "2024-10-16T15:55:59Z"
page_id: 26685
display_title: "CirQ"
---

`<languages />`{=html} Developed by Google, [CirQ](https://quantumai.google/cirq) is an open-source quantum computing library to build, optimize, simulate and run quantum circuits. More specifically, CirQ allows to simulate circuits on particular qubit configurations, which can optimize a circuit for a certain qubit architecture. Information on the features can be found in the CirQ [documentation](https://quantumai.google/cirq) and [GitHub](https://github.com/quantumlib/Cirq). Like [Snowflurry](https://docs.alliancecan.ca/Snowflurry/en "Snowflurry"){.wikilink}, CirQ can be used to run quantum circuits on the [MonarQ](https://docs.alliancecan.ca/MonarQ/en "MonarQ"){.wikilink} quantum computer.

## Installation

The CirQ simulator is available on all of our clusters. To have access, you must load the [Python](https://docs.alliancecan.ca/Python/fr "Python"){.wikilink} language. Il est préférable de travailler dans un [environnement virtuel Python](https://docs.alliancecan.ca/Python/fr#Créer_et_utiliser_un_environnement_virtuel "environnement virtuel Python"){.wikilink}. 1.4.1 \|python -c \"import cirq\" \|pip freeze \> cirq-1.4.1-reqs.txt }} The last command creates the cirq-1.4.1-reqs.txt file which you can also use in a job script such as in the example below.

## Exécution sur une grappe {#exécution_sur_une_grappe}

```{=mediawiki}
{{File
  |name=script.sh
  |lang="sh"
  |contents=
#!/bin/bash
#SBATCH --account=def-someuser # Modify with your account name
#SBATCH --time=00:15:00        # Modify as needed
#SBATCH --cpus-per-task=1      # Modify as needed
#SBATCH --mem-per-cpu=1G       # Modify as needed

# Load modules dependencies.
module load StdEnv/2023 gcc python/3.11 

# Generate your virtual environment in $SLURM_TMPDIR.                                                                                           
virtualenv --no-download ${SLURM_TMPDIR}/env                                                                                                                   
source ${SLURM_TMPDIR}/env/bin/activate  

# Install CirQ and its dependencies.                                                                                                                                                                                                                                                                                  
pip install --no-index --upgrade pip                                                                                                                            
pip install --no-index --requirement ~/cirq-1.4.1-reqs.txt

# Edit with your CirQ program.                                                                                                                                                             
python cirq_example.py
}}
```
You can then [submit your job to the scheduler](https://docs.alliancecan.ca/Running_jobs "submit your job to the scheduler"){.wikilink}.

## Use case: Bell states {#use_case_bell_states}

Les états de Bell sont les états les plus simples qui permettent d\'expliquer à la fois la superposition et l\'intrication sur des qubits. La bibliothèque [CirQ](https://github.com/quantumlib/Cirq) permet de construire un état de Bell comme ceci : `<noinclude>`{=html}

`</noinclude>`{=html} ![](Bell_Circuit_CirQ.png "Bell_Circuit_CirQ.png") This code builds and displays a circuit that prepares a Bell state. The H gate (Hadamard gate) creates an equal superposition of \|0⟩ and \|1⟩ on the first qubit while the CNOT gate (controlled X gate) creates an entanglement between the two qubits. This Bell state is therefore an equal superposition of the states \|00⟩ and \|11⟩. Simulating this circuit using CirQ allows you to visualize the results. In this diagram, the integer 3 represents the state \|11⟩ since 3 is written 11 in binary. `<noinclude>`{=html}

`</noinclude>`{=html} ![](Bell_Graph_CirQ.png "Bell_Graph_CirQ.png")
