---
title: "MonarQ/en"
url: "https://docs.alliancecan.ca/wiki/MonarQ/en"
category: "General"
last_modified: "2025-03-30T13:30:49Z"
page_id: 26558
display_title: "MonarQ"
---

`<languages />`{=html}

  ----------------------------------------
  Availability: January 2025
  Login node: **monarq.calculquebec.ca**
  ----------------------------------------

\'\'

MonarQ is a 24-qubit superconducting quantum computer developed in Montreal by [Anyon Systems](https://anyonsys.com/) and located at the [École de technologie supérieure](http://www.etsmtl.ca/). See section [Technical specifications](https://docs.alliancecan.ca/MonarQ/en#Technical_specifications "Technical specifications"){.wikilink} below.

Its name is inspired by the monarch butterfly, a symbol of evolution and migration. The capital Q denotes the quantum nature of the computer and its origins in Quebec. Acquisition of MonarQ was made possible with the support of the [Ministère de l\'Économie, de l\'Innovation et de l\'Énergie du Québec (MEIE)](https://www.economie.gouv.qc.ca/) and [Canada Economic Development (CED)](https://ced.canada.ca/en/ced-home/).

## Getting access to MonarQ {#getting_access_to_monarq}

1.  To begin the process of getting access to MonarQ, [complete this form](https://forms.gle/zH1a3oB4SGvSjAwh7). It can only be completed by the principal investigator.
2.  You must have an [account with the Alliance](https://alliancecan.ca/en/services/advanced-research-computing/account-management/apply-account) in order to get access to MonarQ.
3.  Meet with our team to discuss the specifics of your project.
4.  Receive access to the MonarQ dashboard and generate your access token.
5.  To get started using MonarQ, see [ Getting started](https://docs.alliancecan.ca/MonarQ/en#Getting_started " Getting started"){.wikilink} below.

Contact our quantum team at <quantique@calculquebec.ca> if you have any questions or if you want to have a more general discussion before requesting access to MonarQ.

## Technical specifications {#technical_specifications}

![MonarQ qubit mapping](https://docs.alliancecan.ca/QPU.png "MonarQ qubit mapping")

Like quantum processors available today, MonarQ operates in an environment where noise remains a significant factor. Performance metrics, updated at each calibration, are accessible via the Thunderhead portal which you will be able to use after being approved for access to MonarQ.

Among the metrics are:

- 24-qubit quantum processor
- Single-qubit gate: 99.8% fidelity with gate duration of 15ns
- Two-qubit gate: 95.6% fidelity with gate duration of 35ns
- Coherence time: 4-10μs (depending on state)
- Maximum circuit depth: approximately 350 for single-qubit gates and 115 for two-qubit gates

## Quantum computing software {#quantum_computing_software}

There are several specialized software libraries for quantum computing and the development of quantum algorithms. These libraries allow you to build circuits that are executed on simulators that mimic the performance and results obtained on a quantum computer such as MonarQ. They can be used on all Alliance clusters.

- [PennyLane](https://docs.alliancecan.ca/PennyLane/en "PennyLane"){.wikilink}, for Python commands
- [Snowflurry](https://docs.alliancecan.ca/Snowflurry/en "Snowflurry"){.wikilink}, for Julia commands
- [Qiskit](https://docs.alliancecan.ca/Qiskit/fr "Qiskit"){.wikilink}, for Python commands

The quantum logic gates of the MonarQ processor are called through a [Snowflurry](https://docs.alliancecan.ca/Snowflurry/en "Snowflurry"){.wikilink} software library written in [Julia](https://docs.alliancecan.ca/Julia "Julia"){.wikilink}. Although MonarQ is natively compatible with Snowflurry, there is a [PennyLane-Snowflurry](https://github.com/calculquebec/pennylane-snowflurry\) plugin developed by Calcul Québec that allows you to execute circuits on MonarQ while benefiting from the features and development environment offered by [PennyLane](https://docs.alliancecan.ca/PennyLane/en "PennyLane"){.wikilink}.

## Getting started {#getting_started}

**Prerequisites**: Make sure you have access to MonarQ and that you have your login credentials (`<i>`{=html}username`</i>`{=html}, `<i>`{=html}API token`</i>`{=html}). If you have any questions, write to <quantique@calculquebec.ca>.

- **Step 1: Connect to [Narval](https://docs.alliancecan.ca/Narval/en "Narval"){.wikilink}**
  - MonarQ is only accessible from Narval, a Calcul Québec cluster. Narval is accessed from the login node **narval.alliancecan.ca**.
  - For help connecting to Narval, see [SSH](https://docs.alliancecan.ca/SSH/en "SSH"){.wikilink}.

<!-- -->

- **Step 2: Create the environment**
  - Create a Python virtual environment (3.11 or later) to use PennyLane and the [PennyLane-CalculQuébec](https://github.com/calculquebec/pennylane-snowflurry\) plugin. These are already installed on Narval so that you will only have to import the software libraries you want.

<!-- -->

- **Step 3: Configure your identifiers on MonarQ and define MonarQ as your device**
  - Open a Python .py file and import the required dependencies (in the following example, PennyLane and MonarqClient).
  - Create a client with your identifiers. Your token is available through the Thunderhead portal. The host is **monarq.calculquebec.ca**.
  - Create a PennyLane device with your client. You can also enter the number of qubits (`<i>`{=html}wires`</i>`{=html}) and the number of shots.
  - For more information, see [pennylane_calculquebec](https://github.com/calculquebec/pennylane-calculquebec/blob/main/doc/getting_started.ipynb).

<!-- -->

- **Step 4: Create your circuit**
  - In the same Python file, you can now code your quantum circuit.

<!-- -->

- **Step 5: Execute your circuit from the scheduler**
  - The [`sbatch`](https://slurm.schedmd.com/sbatch.html) command is used to submit a task.

``` bash
$ sbatch simple_job.sh
Submitted batch job 123456
```

The Slurm script is similar to

- The result is written to a file with a name starting with slurm-, followed by the task ID and the .out suffix, for example `<i>`{=html}slurm-123456.out`</i>`{=html}.
- The file contains the result in dictionary `{'000': 496, '001': 0, '010': 0, '011': 0, '100': 0, '101': 0, '110': 0, '111': 504}`.
- For more information on submitting tasks on Narval, see [Running jobs](https://docs.alliancecan.ca/Running_jobs "Running jobs"){.wikilink}.

## FAQ

- [Foire aux questions (FAQ)](https://docs.google.com/document/d/13sfHwJTo5tcmzCZQqeDmAw005v8I5iFeKp3Xc_TdT3U/edit?tab=t.0)

## Other tools {#other_tools}

- [Quantum transpilation](https://docs.alliancecan.ca/Transpileur_quantique/en "Quantum transpilation"){.wikilink}

## Applications

MonarQ is suited for computations requiring small quantities of high-fidelity qubits, making it an ideal tool to develop and test quantum algorithms. Other possible applications include modelling small quantum systems; testing new methods and techniques for quantum programming and error correction; and more generally, fundamental research in quantum computing.

## Technical support {#technical_support}

For questions about our quantum services, write to <quantique@calculquebec.ca>.\
Sessions on quantum computing and programming with MonarQ are [listed here](https://www.eventbrite.com/o/calcul-quebec-8295332683).\
