---
title: "Qiskit/en"
url: "https://docs.alliancecan.ca/wiki/Qiskit/en"
category: "General"
last_modified: "2026-04-09T13:58:02Z"
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
== Using Qiskit with MonarQ ==

You can use MonarQ directly with Qiskit via the qiskit-calculquebec plugin. This plugin allows you to develop and run Qiskit circuits on the Calcul Québec infrastructure.

=== Install the dependencies ===

* Step 1: Install the dependencies

* Note: qiskit-calculquebec installs Qiskit automatically.

=== MonarQ backend initialisation ===

* Step 2: Set up your credentials and the backend
** Create a client using your credentials. Your token is available via the Thunderhead portal.
** The host is ‘’'https://monarq.calculquebec.ca.‘’'
** Then initialize the MonarQ backend.

=== Running the circuit ===

* Step 3: Transpile and run the circuit

=== Notes ===

* Transpilation is required to adapt the circuit to MonarQ's native connectivity and ports.
* The number of shots can be adjusted as needed (maximum: 1024).
* The use of ‘’'SamplerV2'‘’ is recommended for running circuits with measures.