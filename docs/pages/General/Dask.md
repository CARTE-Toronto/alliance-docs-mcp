---
title: "Dask/en"
url: "https://docs.alliancecan.ca/wiki/Dask/en"
category: "General"
last_modified: "2025-05-30T08:34:16Z"
page_id: 27324
display_title: "Dask"
---

`<languages />`{=html} [Dask](https://docs.dask.org/en/stable/) is a flexible library for parallel computing in Python. It provides distributed NumPy array and Pandas DataFrame objects, as well as enabling distributed computing in pure Python with access to the PyData stack.

# Installing our wheel {#installing_our_wheel}

The preferred option is to install it using our provided Python [wheel](https://pythonwheels.com/) as follows:

:   1\. Load a Python [module](https://docs.alliancecan.ca/Utiliser_des_modules/en#Sub-command_load "module"){.wikilink}, thus `module load python/3.11`
:   2\. Create and start a [virtual environment](https://docs.alliancecan.ca/Python#Creating_and_using_a_virtual_environment "virtual environment"){.wikilink}.
:   3\. Install `dask`, and optionally `dask-distributed` in the virtual environment with `pip install`.

<!-- -->

:   

# Job submission {#job_submission}

## Single node {#single_node}

Below is an example of a job that spawns a single-node Dask cluster with 6 cpus and computes the mean of a column of a parallelized dataframe.

In the script `Dask-example.py`, we launch a Dask cluster with as many worker processes as there are cores in our job. This means each worker will spawn at most one CPU thread. For a complete discussion of how to reason about the number of worker processes and the number of threads per worker, see the [official Dask documentation](https://distributed.dask.org/en/stable/efficiency.html?highlight=workers%20threads#adjust-between-threads-and-processes). In this example, we split a pandas data frame into 6 chunks, so each worker will process a part of the data frame using one CPU: `{{File
  |name=dask-example.py
  |lang="python"
  |contents=
import pandas as pd

from dask import dataframe as dd
from dask.distributed import Client

import os

n_workers = int(os.environ['SLURM_CPUS_PER_TASK'])

client = Client(f"tcp://{os.environ['DASK_SCHEDULER_ADDR']}:{os.environ['DASK_SCHEDULER_PORT']}")

index = pd.date_range("2021-09-01", periods=2400, freq="1H")
df = pd.DataFrame({"a": np.arange(2400)}, index=index)
ddf = dd.from_pandas(df, npartitions=n_workers) # split the pandas data frame into "n_workers" chunks

result = ddf.a.mean().compute()

print(f"The mean is {result}")

}}`{=mediawiki}

## Multiple nodes {#multiple_nodes}

In the example that follows, we reproduce the single-node example, but this time with a two-node Dask cluster, with 6 CPUs on each node. This time we also spawn 2 workers per node, each with 3 cores.

Where the script `config_virtualenv.sh` is: `{{File
  |name=config_env.sh
  |lang="bash"
  |contents=
#!/bin/bash

echo "From node ${SLURM_NODEID}: installing virtualenv..."

module load python gcc arrow
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index dask[distributed,dataframe]

echo "Done installing virtualenv!"

deactivate
}}`{=mediawiki} And the script `launch_dask_workers.sh` is:

And, finally, the script `test_dask.py` is: `{{File
  |name=test_dask.py
  |lang="python"
  |contents=
import pandas as pd
import numpy as np

from dask import dataframe as dd
from dask.distributed import Client

import os

client = Client(f"tcp://{os.environ['DASK_SCHEDULER_ADDR']}:{os.environ['DASK_SCHEDULER_PORT']}")

index = pd.date_range("2021-09-01", periods=2400, freq="1H")
df = pd.DataFrame({"a": np.arange(2400)}, index=index)
ddf = dd.from_pandas(df, npartitions=6)

result = ddf.a.mean().compute()

print(f"The mean is {result}")

}}`{=mediawiki}
