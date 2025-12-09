---
title: "Dedalus/en"
url: "https://docs.alliancecan.ca/wiki/Dedalus/en"
category: "General"
last_modified: "2024-09-30T17:25:50Z"
page_id: 25399
display_title: "Dedalus"
---

__FORCETOC__

Dedalus is a flexible framework for solving partial differential equations using modern spectral methods.

= Available versions =
Dedalus is available on our clusters as prebuilt Python packages (wheels). You can list available versions with avail_wheels.

= Installing Dedalus in a Python virtual environment =
1. Load Dedalus runtime dependencies.

2. Create and activate a Python virtual environment.

3. Install a specific version of Dedalus and its Python dependencies.
X.Y.Z
}}
where X.Y.Z is the exact desired version, for instance 3.0.2.
You can omit to specify the version in order to install the latest one available from the wheelhouse.

4. Validate it.

5. Freeze the environment and requirements set.

6. Remove the local virtual environment.

= Running Dedalus =
You can run Dedalus distributed across multiple nodes or cores.
For efficient MPI scheduling, please see:
* MPI job
* Advanced MPI scheduling

1. Write your job submission script.

2. Submit your job to the scheduler.

Before submitting your job, it is important to test that your submission script will start without errors.
You can do a quick test in an interactive job.