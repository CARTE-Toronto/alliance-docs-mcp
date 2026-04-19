---
title: "Advanced Job Submission/en"
url: "https://docs.alliancecan.ca/wiki/Advanced_Job_Submission/en"
category: "General"
last_modified: "2026-04-13T17:26:02Z"
page_id: 32846
display_title: "Advanced Job Submission"
---

== Managing numerous compute tasks ==

The following tools are helpful when you need to process multiple files with or without different parameter combinations (parameter sweep):

* Job Arrays: to submit several compute tasks in one single script, an ideal method when each job exceeds one hour and the number of jobs is under one thousand;
* GNU Parallel: to run and manage several short compute tasks, including parameter sweeps, on a single node reserved via a parallel job;
* GLOST: the Greedy Launcher Of Small Tasks uses MPI and a manager-worker architecture to progressively run a long list of serial tasks on CPU cores reserved via a parallel job;
* META: a suite of scripts designed at SHARCNET to automate high-throughput computing (running a large number of related serial, parallel, or GPU calculations).

== Inter-job dependencies ==

While Slurm jobs are the building blocks for compute pipelines, inter-job dependencies are the links and relationships between the steps of a pipeline. For example, if two different jobs need to run one after the other, the second job depends on the first one. The second could depend on the start time, the end time or the final status of the first job. Typically, we want the second job to be started only once the first job has succeeded. For example:

$(sbatch --parsable job1.sh)           # Save the first job ID
|sbatch --dependencyafterok:$JOBID1 job2.sh   # Depends on the first job
}}

Notes
* Multiple jobs can have the same dependency (multiple jobs waiting for one job).
* A job can have multiple dependencies (one job waiting for multiple jobs).
* There are multiple types of dependencies: after, afterany, afterok, afternotok, etc. For more details, see the --dependency option on the official sbatch documentation page.

== Heterogeneous jobs ==

The Slurm scheduler supports heterogeneous jobs. This could be very useful if you know in advance that your MPI application will require more CPU cores and more memory for the main process than for the other processes.

For example, if the main process requires 8 cores and a total of 32GB of RAM, while the other processes only require 1 core and 1GB of RAM, we can specify both types of requirements in a job script:

Or we can separate resource requests with a colon (:) on the sbatch command line:

1 --cpus-per-task8 --mem-per-cpu4000M : --ntasks15 --cpus-per-task1 --mem-per-cpu1000M  mpi_job.sh
}}