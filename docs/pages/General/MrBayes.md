---
title: "MrBayes/en"
url: "https://docs.alliancecan.ca/wiki/MrBayes/en"
category: "General"
last_modified: "2025-07-23T15:18:18Z"
page_id: 25953
display_title: "MrBayes"
---

MrBayes is a program for Bayesian inference and model choice across a wide range of phylogenetic and evolutionary models. MrBayes uses Markov chain Monte Carlo (MCMC) methods to estimate the posterior distribution of model parameters.

== Finding available modules ==

For more on finding and selecting a version of MrBayes using module commands see Using modules

== Examples ==

=== Sequential ===
The following job script uses only one CPU core (--cpus-per-task=1).
The example uses an input file (primates.nex) distributed with MrBayes.

The job script can be submitted with

=== Parallel ===
MrBayes can be run on multiple cores, on multiple nodes, and on GPUs.

==== MPI ====
The following job script will use 8 CPU cores in total, on one or more nodes.
Like the previous example, it uses an input file (primates.nex) distributed with MrBayes.

The job script can be submitted with

==== GPU ====
The following job script will use a GPU.
Like the previous examples, it uses an input file (primates.nex) distributed with MrBayes.

The job script can be submitted with

== Checkpointing ==
If you need very long runs of MrBayes, we suggest you break up the work into several small jobs rather than one very long job. Long jobs have are more likely to be interrupted by hardware failure or maintenance outage. Fortunately, MrBayes has a mechanism for creating checkpoints, in which progress can be saved from one job and continued in a subsequent job.

Here is an example of how to split a calculation into two Slurm jobs which will run one after the other.  Create two files, job1.nex and job2.nex, as shown below.  Notice that the key difference between them is the presence of the append keyword in the second.

Then create a job script.  This example is a job array, which means that one script and
one sbatch command will be sufficient to launch two Slurm jobs, and therefore
both parts of the calculation.  See Job arrays for more about the --array
parameter and the $SLURM_ARRAY_TASK_ID variable used here.

The example can be submitted with