---
title: "Qiskit/en"
url: "https://docs.alliancecan.ca/wiki/Qiskit/en"
category: "General"
last_modified: "2026-03-27T18:18:35Z"
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