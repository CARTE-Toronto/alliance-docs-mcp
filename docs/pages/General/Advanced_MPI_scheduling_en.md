---
title: "Advanced MPI scheduling/en"
url: "https://docs.alliancecan.ca/wiki/Advanced_MPI_scheduling/en"
category: "General"
last_modified: "2025-11-14T22:00:43Z"
page_id: 3424
display_title: "Advanced MPI scheduling"
---

`<languages />`{=html}

Most users should submit MPI or distributed memory parallel jobs following the example given at [Running jobs](https://docs.alliancecan.ca/Running_jobs#MPI_job "wikilink"). Simply request a number of processes with `--ntasks` or `-n` and trust the scheduler to allocate those processes in a way that balances the efficiency of your job with the overall efficiency of the cluster.

If you want more control over how your job is allocated, then SchedMD\'s page on [multicore support](https://slurm.schedmd.com/mc_support.html) is a good place to begin. It describes how many of the options to the [`sbatch`](https://slurm.schedmd.com/sbatch.html) command interact to constrain the placement of processes.

You may find this discussion on [What exactly is considered a CPU?](https://slurm.schedmd.com/faq.html#cpu_count) in Slurm to be useful.

### Examples of common MPI scenarios {#examples_of_common_mpi_scenarios}

#### Few cores, any number of nodes {#few_cores_any_number_of_nodes}

In addition to the time limit needed for `<i>`{=html}any`</i>`{=html} Slurm job, an MPI job requires that you specify how many MPI processes Slurm should start. The simplest way to do this is with `--ntasks`. Since the default memory allocation of 256MB per core is often insufficient, you may also wish to specify how much memory is needed. Using `--ntasks` you cannot know in advance how many cores will reside on each node, so you should request memory with `--mem-per-cpu`. For example:

This will run 15 MPI processes. The cores could be allocated on one node, on 15 nodes, or on any number in between.

#### Whole nodes {#whole_nodes}

If you have a large parallel job to run, that is, one that can efficiently use 64 cores or more, you should probably request whole nodes. To do so, it helps to know what node types are available at the cluster you are using.

Typical nodes in [Fir](https://docs.alliancecan.ca/Fir "wikilink"), [Narval](https://docs.alliancecan.ca/Narval/en "wikilink"), [Nibi](https://docs.alliancecan.ca/Nibi "wikilink"), [Rorqual](https://docs.alliancecan.ca/Rorqual/en "wikilink") and [Trillium](https://docs.alliancecan.ca/Trillium "wikilink") have the following CPU and memory configuration:

  Cluster                            cores   usable memory   Notes
  ---------------------------------- ------- --------------- ----------------------------------------------------
  [Fir](https://docs.alliancecan.ca/Fir "wikilink")              192     750 GiB         Some are reserved for whole node jobs.
  [Narval](https://docs.alliancecan.ca/Narval/en "wikilink")     64      249 GiB         Some are reserved for whole node jobs.
  [Nibi](https://docs.alliancecan.ca/Nibi "wikilink")            192     748 GiB         No node specifically reserved for whole node jobs.
  [Rorqual](https://docs.alliancecan.ca/Rorqual/en "wikilink")   192     750 GiB         Some are reserved for whole node jobs.
  [Trillium](https://docs.alliancecan.ca/Trillium "wikilink")    192     749 GiB         Only whole-node jobs are possible at Trillium.

Whole-node jobs are allowed to run on any node. In the table above, `<i>`{=html}Some are reserved for whole-node jobs`</i>`{=html} indicates that there are nodes on which by-core jobs are forbidden.

A job script requesting whole nodes should look like this:

`<tabs>`{=html} `<tab name="Fir">`{=html} `</tab>`{=html} `<tab name="Narval">`{=html} `</tab>`{=html} `<tab name="Nibi">`{=html} `</tab>`{=html} `<tab name="Rorqual">`{=html} `</tab>`{=html} `<tab name="Trillium">`{=html} `</tab>`{=html} `</tabs>`{=html}

Requesting `--mem=0` is interpreted by Slurm to mean `<i>`{=html}reserve all the available memory on each node assigned to the job.`</i>`{=html}

If you need more memory per node than the smallest node provides (e.g. more than 748 GiB at Nibi) then you `<b>`{=html}should not`</b>`{=html} use `--mem=0`, but request the amount explicitly. Furthermore, some memory on each node is reserved for the operating system. To find the largest amount your job can request and still qualify for a given node type, see the `<i>`{=html}Available memory`</i>`{=html} column of the `<i>`{=html}Node characteristics`</i>`{=html} table for each cluster.

-   [Fir node characteristics](https://docs.alliancecan.ca/Fir#Node_characteristics "wikilink")
-   [Narval node characteristics](https://docs.alliancecan.ca/Narval/en#Node_characteristics "wikilink")
-   [Nibi node characteristics](https://docs.alliancecan.ca/Nibi#Node_characteristics "wikilink")
-   [Rorqual node characteristics](https://docs.alliancecan.ca/Rorqual/en#Node_characteristics "wikilink")

#### Few cores, single node {#few_cores_single_node}

If you need less than a full node but need all the cores to be on the same node, then you can request, for example,

In this case, you could also say `--mem-per-cpu=3G`. The advantage of `--mem=45G` is that the memory consumed by each individual process doesn\'t matter, as long as all of them together don't use more than 45GB. With `--mem-per-cpu=3G`, the job will be cancelled if any of the processes exceeds 3GB.

#### Large parallel job, not a multiple of whole nodes {#large_parallel_job_not_a_multiple_of_whole_nodes}

Not every application runs with maximum efficiency on a multiple of 32 (or 40, or 48) cores. Choosing the number of cores to request---and whether or not to request whole nodes---may be a trade-off between `<i>`{=html}running`</i>`{=html} time (or efficient use of the computer) and `<i>`{=html}waiting`</i>`{=html} time (or efficient use of your time). If you want help evaluating these factors, please contact [Technical support](https://docs.alliancecan.ca/Technical_support "wikilink").

### Hybrid jobs: MPI and OpenMP, or MPI and threads {#hybrid_jobs_mpi_and_openmp_or_mpi_and_threads}

It is important to understand that the number of `<i>`{=html}tasks`</i>`{=html} requested of Slurm is the number of `<i>`{=html}processes`</i>`{=html} that will be started by `srun`. So for a hybrid job that will use both MPI processes and OpenMP threads or Posix threads, you should set the MPI process count with `--ntasks` or `-ntasks-per-node`, and set the thread count with `--cpus-per-task`.

`--ntasks=16`\
`--cpus-per-task=4`\
`--mem-per-cpu=3G`\
`srun --cpus-per-task=$SLURM_CPUS_PER_TASK application.exe`

In this example, a total of 64 cores will be allocated, but only 16 MPI processes (tasks) can and will be initialized. If the application is also OpenMP, then each process will spawn 4 threads, one per core. Each process will be allocated with 12GB of memory. The tasks, with 4 cores each, could be allocated anywhere, from 2 to up to 16 nodes. Note that you must specify `--cpus-per-task=$SLURM_CPUS_PER_TASK` for `srun` as well, as this is a requirement since Slurm 22.05 and does not hurt for older versions.

`--nodes=2`\
`--ntasks-per-node=8`\
`--cpus-per-task=4`\
`--mem=96G`\
`srun --cpus-per-task=$SLURM_CPUS_PER_TASK application.exe`

This job is the same size as the last one: 16 tasks (that is, 16 MPI processes), each with 4 threads. The difference here is that we are sure of getting exactly 2 whole nodes. Remember that `--mem` requests memory `<i>`{=html}per node`</i>`{=html}, so we use it instead of `--mem-per-cpu` for the reason described earlier.

### Why srun instead of mpiexec or mpirun? {#why_srun_instead_of_mpiexec_or_mpirun}

`mpirun` is a wrapper that enables communication between processes running on different machines. Modern schedulers already provide many things that `mpirun` needs. With Torque/Moab, for example, there is no need to pass to `mpirun` the list of nodes on which to run, or the number of processes to launch; this is done automatically by the scheduler. With Slurm, the task affinity is also resolved by the scheduler, so there is no need to specify things like

`mpirun --map-by node:pe=4 -n 16  application.exe`

As implied in the examples above, `srun application.exe` will automatically distribute the processes to precisely the resources allocated to the job.

In programming terms, `srun` is at a higher level of abstraction than `mpirun`. Anything that can be done with `mpirun` can be done with `srun`, and more. It is the tool in Slurm to distribute any kind of computation. It replaces Torque's `pbsdsh`, for example, and much more. Think of `srun` as the SLURM `<i>`{=html}all-around parallel-tasks distributor`</i>`{=html}; once a particular set of resources is allocated, the nature of your application doesn\'t matter (MPI, OpenMP, hybrid, serial farming, pipelining, multiprogram, etc.), you just have to `srun` it.

Also, as you would expect, `srun` is fully coupled to Slurm. When you `srun` an application, a `<i>`{=html}job step`</i>`{=html} is started, the environment variables `SLURM_STEP_ID` and `SLURM_PROCID` are initialized correctly, and correct accounting information is recorded.

For an example of some differences between `srun` and `mpiexec`, see [this discussion](https://mail-archive.com/users@lists.open-mpi.org/msg31874.html) on the Open MPI support forum. Better performance might be achievable with `mpiexec` than with `srun` under certain circumstances, but using `srun` minimizes the risk of a mismatch between the resources allocated by Slurm and those used by Open MPI.

### External links {#external_links}

-   [sbatch](https://slurm.schedmd.com/sbatch.html) documentation
-   [srun](https://slurm.schedmd.com/srun.html) documentation
-   [Open MPI](https://www.open-mpi.org/faq/?category=slurm) and Slurm
