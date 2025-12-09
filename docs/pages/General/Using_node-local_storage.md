---
title: "Using node-local storage/en"
url: "https://docs.alliancecan.ca/wiki/Using_node-local_storage/en"
category: "General"
last_modified: "2025-08-29T13:04:31Z"
page_id: 14251
display_title: "Using node-local storage"
---

`<languages />`{=html}

When [Slurm](https://docs.alliancecan.ca/Running_jobs "Slurm"){.wikilink} starts a job, it creates a temporary directory on each node assigned to the job. It then sets the full path name of that directory in an environment variable called `SLURM_TMPDIR`.

Because this directory resides on local disk, input and output (I/O) to it is almost always faster than I/O to a [network storage](https://docs.alliancecan.ca/Storage_and_file_management "network storage"){.wikilink} (/project, /scratch, or /home). Specifically, local disk is better for frequent small I/O transactions than network storage. Any job doing a lot of input and output (which is most jobs!) may expect to run more quickly if it uses `$SLURM_TMPDIR` instead of network storage.

The temporary character of `$SLURM_TMPDIR` makes it more trouble to use than network storage. Input must be copied from network storage to `$SLURM_TMPDIR` before it can be read, and output must be copied from `$SLURM_TMPDIR` back to network storage before the job ends to preserve it for later use.

# Input

In order to `<i>`{=html}read`</i>`{=html} data from `$SLURM_TMPDIR`, you must first copy the data there. In the simplest case, you can do this with `cp` or `rsync`:

    cp /project/def-someone/you/input.files.* $SLURM_TMPDIR/

This may not work if the input is too large, or if it must be read by processes on different nodes. See [Multinode jobs](https://docs.alliancecan.ca/Using_node-local_storage#Multinode_jobs "Multinode jobs"){.wikilink} and [Amount of space`</i>`{=html}](https://docs.alliancecan.ca/Using_node-local_storage#Amount_of_space "Amount of space"){.wikilink} below for more.

## Executable files and libraries {#executable_files_and_libraries}

A special case of input is the application code itself. In order to run the application, the shell started by Slurm must open at least an application file, which it typically reads from network storage. But few applications these days consist of exactly one file; most also need several other files (such as libraries) in order to work.

We particularly find that using an application in a [Python](https://docs.alliancecan.ca/Python "Python"){.wikilink} virtual environment generates a large number of small I/O transactions---more than it takes to create the virtual environment in the first place. This is why we recommend [creating virtual environments inside your jobs](https://docs.alliancecan.ca/Python#Creating_virtual_environments_inside_of_your_jobs "creating virtual environments inside your jobs"){.wikilink} using `$SLURM_TMPDIR`.

# Output

Output data must be copied from `$SLURM_TMPDIR` back to some permanent storage before the job ends. If a job times out, then the last few lines of the job script might not be executed. This can be addressed three ways:

- request enough runtime to let the application finish, although we understand that this isn\'t always possible;
- write [checkpoints](https://docs.alliancecan.ca/Points_de_contrôle/en "checkpoints"){.wikilink} to network storage, not to `$SLURM_TMPDIR`;
- write a signal trapping function.

## Signal trapping {#signal_trapping}

You can arrange that Slurm will send a signal to your job shortly before the runtime expires, and that when that happens your job will copy your output from `$SLURM_TMPDIR` back to network storage. This may be useful if your runtime estimate is uncertain, or if you are chaining together several Slurm jobs to complete a long calculation.

To do so you will need to write a shell function to do the copying, and use the `trap` shell command to associate the function with the signal. See [this page](https://services.criann.fr/en/services/hpc/cluster-myria/guide/signals-sent-by-slurm/) from CRIANN for an example script and detailed guidance.

This method will not preserve the contents of `$SLURM_TMPDIR` in the case of a node failure, or certain malfunctions of the network file system.

# Multinode jobs {#multinode_jobs}

If a job spans multiple nodes and some data is needed on every node, then a simple `cp` or `tar -x` will not suffice.

## Copy files {#copy_files}

Copy one or more files to the `SLURM_TMPDIR` directory on every node allocated like this: \$SLURM_NNODES \--ntasks-per-node1 cp file \[files\...\] \$SLURM_TMPDIR}}

## Compressed archives {#compressed_archives}

### ZIP

Extract to the `SLURM_TMPDIR`: \$SLURM_NNODES \--ntasks-per-node1 unzip archive.zip -d \$SLURM_TMPDIR}}

### Tarball

Extract to the `SLURM_TMPDIR`: \$SLURM_NNODES \--ntasks-per-node1 tar -xvf archive.tar.gz -C \$SLURM_TMPDIR}}

# Amount of space {#amount_of_space}

At `<b>`{=html}[Trillium](https://docs.alliancecan.ca/Trillium "Trillium"){.wikilink}`</b>`{=html}, \$SLURM_TMPDIR is implemented as `<i>`{=html}RAMdisk`</i>`{=html}, so the amount of space available is limited by the memory on the node, less the amount of RAM used by your application.

At the general-purpose clusters, the amount of space available depends on the cluster and the node to which your job is assigned.

  cluster                                      space in \$SLURM_TMPDIR   size of disks
  -------------------------------------------- ------------------------- ---------------
  [Fir](https://docs.alliancecan.ca/Fir "Fir"){.wikilink}                  7T                        7.84T
  [Narval](https://docs.alliancecan.ca/Narval "Narval"){.wikilink}         800G                      960G, 3.84T
  [Nibi](https://docs.alliancecan.ca/Nibi "Nibi"){.wikilink}               3T                        3T, 11T
  [Rorqual](https://docs.alliancecan.ca/Rorqual/en "Rorqual"){.wikilink}   375G                      480G, 3.84T

If your job reserves [whole nodes](https://docs.alliancecan.ca/Advanced_MPI_scheduling#Whole_nodes "whole nodes"){.wikilink}, then you can reasonably assume that this much space is available to you in \$SLURM_TMPDIR on each node. However, if the job requests less than a whole node, then other jobs may also write to the same filesystem (but a different directory!), reducing the space available to your job.

Some nodes at each site have more local disk than shown above. See `<i>`{=html}Node characteristics`</i>`{=html} at the appropriate cluster\'s page ([Fir](https://docs.alliancecan.ca/Fir "Fir"){.wikilink}, [Narval](https://docs.alliancecan.ca/Narval/en "Narval"){.wikilink}, [Nibi](https://docs.alliancecan.ca/Nibi "Nibi"){.wikilink}, [Rorqual](https://docs.alliancecan.ca/Rorqual/en "Rorqual"){.wikilink}) for guidance.
