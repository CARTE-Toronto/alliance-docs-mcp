---
title: "Dask/en"
url: "https://docs.alliancecan.ca/wiki/Dask/en"
category: "General"
last_modified: "2026-01-28T14:35:21Z"
page_id: 27324
display_title: "Dask"
---

Dask is a flexible library for parallel computing in Python. It provides distributed NumPy array and Pandas DataFrame objects, as well as enabling distributed computing in pure Python with access to the PyData stack.

=Installing our wheel=

The preferred option is to install it using our provided Python wheel as follows:
:1. Load a Python module, thus module load python/3.11
:2. Create and start a virtual environment.
:3. Install dask, and optionally dask-distributed in the virtual environment with pip install.

:

=Job submission=

== Single node ==
Below is an example of a job that spawns a single-node Dask cluster with 6 cpus and computes the mean of a column of a parallelized dataframe.

In the script Dask-example.py, we launch a Dask cluster with as many worker processes as there are cores in our job. This means each worker will spawn at most one CPU thread. For a complete discussion of how to reason about the number of worker processes and the number of threads per worker, see the official Dask documentation. In this example, we split a pandas data frame into 6 chunks, so each worker will process a part of the data frame using one CPU:

== Multiple nodes ==
In the example that follows, we reproduce the single-node example, but this time with a two-node Dask cluster, with 6 CPUs on each node. This time we also spawn 2 workers per node, each with 3 cores.

Where the script config_virtualenv.sh is:

And the script launch_dask_workers.sh is:

And, finally, the script test_dask.py is: