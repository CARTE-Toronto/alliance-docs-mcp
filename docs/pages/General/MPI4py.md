---
title: "MPI4py/en"
url: "https://docs.alliancecan.ca/wiki/MPI4py/en"
category: "General"
last_modified: "2024-10-17T19:25:01Z"
page_id: 26625
display_title: "MPI4py"
---

MPI for Python provides Python bindings for the Message Passing Interface (MPI) standard, allowing Python applications to exploit multiple processors on workstations, clusters and supercomputers.
__FORCETOC__

= Available versions =
mpi4py is available as a module, and not from the wheelhouse as typical Python packages are.
You can find available version with

and look for more information on a specific version with

where X.Y.Z is the exact desired version, for instance 4.0.0.

= Famous first words: Hello World =
1. Run a short interactive job.
 --ntasks5}}

2. Load the module.

3. Run a Hello World test.

In the case above, two nodes (node1 and node3) were allocated, and the jobs were distributed across the available resources.

= mpi4py as a package dependency =
Often mpi4py is a dependency of another package. In order to fulfill this dependency :

1. Deactivate any Python virtual environment.

Note: If you had a virtual environment activated, it is important to deactivate it first, then load the module, before reactivating your virtual environment.

2. Load the module.

3. Check that it is visible by pip
 grep mpi4py
|result=
mpi4py            4.0.0
}}
and is accessible for your currently loaded python module.

If no errors are raised, then everything is OK!

4. Create a virtual environment and install your packages.

= Running jobs =
You can run mpi jobs distributed across multiple nodes or cores.
For efficient MPI scheduling, please see:
* MPI job
* Advanced MPI scheduling

== CPU ==
1. Write your python code, for instance, broadcasting a numpy array.

The example above is based on the mpi4py tutorial.

2. Write your submission script.

3. Test your script.

Before submitting your job, it is important to test that your submission script will start without errors. You can do a quick test in an interactive job.

4. Submit your job to the scheduler.

== GPU ==
1. From a login node, download the demo example.

The example above and others, can be found in the demo folder.

2. Write your submission script.

3. Test your script.

Before submitting your job, it is important to test that your submission script will start without errors.
You can do a quick test in an interactive job.

4. Submit your job

= Troubleshooting =

== ModuleNotFoundError: No module named 'mpi4py' ==
If mpi4py is not accessible, you may get the following error when importing it:

ModuleNotFoundError: No module named 'mpi4py'

Possible solutions:
* check which Python versions are compatible with your loaded mpi4py module using module spider mpi4py/X.Y.Z. Once a compatible Python module is loaded, check that python -c 'import mpi4py' works.
* load the module before activating your virtual environment: please see the mpi4py as a package dependency section above.

See also ModuleNotFoundError: No module named 'X'.