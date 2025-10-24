---
title: "Advanced Job Submission"
url: "https://docs.alliancecan.ca/wiki/Advanced_Job_Submission"
category: "General"
last_modified: "2021-11-04T18:52:45Z"
page_id: 13946
display_title: "Advanced Job Submission"
---

## Submitting Numerous Compute Tasks {#submitting_numerous_compute_tasks}

In case you have multiple data files to process with many different combinations of parameters, you should not have to create and submit thousands of job scripts to the scheduler. Here are three proposed tools to answer this kind of need:

- [GNU Parallel](https://docs.alliancecan.ca/GNU_Parallel "GNU Parallel"){.wikilink}: fill an entire compute node with serial compute tasks addressing multiple combinations of arguments.
- [Job Arrays](https://docs.alliancecan.ca/Job_arrays "Job Arrays"){.wikilink}: a single script that represents multiple Slurm jobs, each of them identified by an integer value.
- [GLOST](https://docs.alliancecan.ca/GLOST "GLOST"){.wikilink}: the Greedy Launcher Of Small Tasks is a kind of sophisticated mix of both above tools.
- [META](https://docs.alliancecan.ca/META:_A_package_for_job_farming "META"){.wikilink}: a suite of scripts designed in SHARCNET to automate high-throughput computing (running a large number of related serial, parallel, or GPU calculations).

## Inter-job Dependencies {#inter_job_dependencies}

While Slurm jobs are building-blocks of pipelines, inter-job dependencies are the links and relationships between each step of pipelines. For example, if two different jobs need to run one after the other, the second job *depends* on the first one. The dependency could depend on the start time, the end time or the final status of the first job. Typically, we want the second job to be started only once the first job has succeeded:

`JOBID1=$(sbatch --parsable job1.sh)           # Save the first job ID`\
`sbatch --dependency=afterok:$JOBID1 job2.sh   # Submit a job with a dependency to the first job`

Note:

- Multiple jobs can have the same dependency (multiple jobs are waiting after one job)
- A job can have multiple dependencies (one job is waiting after multiple jobs)
- There are multiple types of dependencies: `after`, `afterany`, `afterok`, `afternotok`, etc. For more details, please look for the `--dependency` option on the [official Sbatch documentation page](https://slurm.schedmd.com/sbatch.html).

## Heterogeneous jobs {#heterogeneous_jobs}

The Slurm scheduler supports [heterogeneous jobs](https://slurm.schedmd.com/heterogeneous_jobs.html). This could be very useful if you know in advance that your [MPI](https://docs.alliancecan.ca/MPI "MPI"){.wikilink} application will require more cores and more memory for the main process than for other processes.

For example, if the main process requires 5 cores and 16GB of RAM, while other processes only require 1 core and 1GB of RAM, we can specify both types of requirements in a job script:

Or we can separate resource requests with a colon (`:`) on the `sbatch` commande line:

`sbatch --ntasks=1 --cpus-per-task=5 --mem-per-cpu=16000M : --ntasks=15 --cpus-per-task=1 --mem-per-cpu=1000M  mpi_job.sh`
