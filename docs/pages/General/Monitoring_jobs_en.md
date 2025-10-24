---
title: "Monitoring jobs/en"
url: "https://docs.alliancecan.ca/wiki/Monitoring_jobs/en"
category: "General"
last_modified: "2025-05-08T19:22:05Z"
page_id: 28553
display_title: "Monitoring jobs"
---

`<languages/>`{=html}

Ensuring that your jobs make efficient use of the resources that are assigned to them is an important part of being a responsible user. This is particularly true when you are using a new program or have made some other substantial change in the work being done by your job. This page describes a variety of different methods for evaluating the efficiency of jobs, both currently running jobs and those which have finished.

### Current jobs {#current_jobs}

By default [squeue](https://slurm.schedmd.com/squeue.html) will show all the jobs the scheduler is managing at the moment. It will run much faster if you ask only about your own jobs with

`$ squeue -u $USER`

You can also use the utility `sq` to do the same thing with less typing.

You can show only running jobs, or only pending jobs:

`$ squeue -u ``<username>`{=html}` -t RUNNING`\
`$ squeue -u ``<username>`{=html}` -t PENDING`

You can show detailed information for a specific job with [scontrol](https://slurm.schedmd.com/scontrol.html):

`$ scontrol show job -dd ``<jobid>`{=html}

`<b>`{=html}Do not`</b>`{=html} run `squeue` from a script or program at high frequency (e.g., every few seconds). Responding to `squeue` adds load to Slurm and may interfere with its performance or correct operation.

#### Email notification {#email_notification}

You can ask to be notified by email of certain job conditions by supplying options to sbatch:

`#SBATCH --mail-user=your.email@example.com`\
`#SBATCH --mail-type=ALL`

For a complete list of the options see [SchedMD\'s documentation](https://slurm.schedmd.com/sbatch.html#OPT_mail-type).

#### Output buffering {#output_buffering}

Output from a non-interactive Slurm job is normally `<i>`{=html}buffered`</i>`{=html}, which means that there is usually a delay between when data is written by the job and when you can see the output on a login node. Depending on the application you are running and the load on the filesystem, this delay can range from less than a second to many minutes, or until the job completes.

There are methods to reduce or eliminate the buffering, but we do not recommend using them because buffering is vital to preserving the overall performance of the filesystem. If you need to monitor the output from a job in `<i>`{=html}real time`</i>`{=html}, we recommend you run an [interactive job](https://docs.alliancecan.ca/#Interactive_jobs "interactive job"){.wikilink} as described above.

### Completed jobs {#completed_jobs}

Get a short summary of the CPU and memory efficiency of a job with `seff`:

`$ seff 12345678`\
`Job ID: 12345678`\
`Cluster: cedar`\
`User/Group: jsmith/jsmith`\
`State: COMPLETED (exit code 0)`\
`Cores: 1`\
`CPU Utilized: 02:48:58`\
`CPU Efficiency: 99.72% of 02:49:26 core-walltime`\
`Job Wall-clock time: 02:49:26`\
`Memory Utilized: 213.85 MB`\
`Memory Efficiency: 0.17% of 125.00 GB`

Find more detailed information about a completed job with [sacct](https://slurm.schedmd.com/sacct.html), and optionally, control what it prints using `--format`:

`$ sacct -j ``<jobid>`{=html}\
`$ sacct -j ``<jobid>`{=html}` --format=JobID,JobName,MaxRSS,Elapsed`

The output from `sacct` typically includes records labelled `.bat+` and `.ext+`, and possibly `.0, .1, .2, ...`. The batch step (`.bat+`) is your submission script - for many jobs that\'s where the main part of the work is done and where the resources are consumed. If you use `srun` in your submission script, that would create a `.0` step that would consume most of the resources. The extern (`.ext+`) step is basically prologue and epilogue and normally doesn\'t consume any significant resources.

If a node fails while running a job, the job may be restarted. `sacct` will normally show you only the record for the last (presumably successful) run. If you wish to see all records related to a given job, add the `--duplicates` option.

Use the MaxRSS accounting field to determine how much memory a job needed. The value returned will be the largest [resident set size](https://en.wikipedia.org/wiki/Resident_set_size) for any of the tasks. If you want to know which task and node this occurred on, print the MaxRSSTask and MaxRSSNode fields also.

The [sstat](https://slurm.schedmd.com/sstat.html) command works on a running job much the same way that [sacct](https://slurm.schedmd.com/sacct.html) works on a completed job.

### Attaching to a running job {#attaching_to_a_running_job}

It is possible to connect to the node running a job and execute new processes there. You might want to do this for troubleshooting or to monitor the progress of a job.

Suppose you want to run the utility [`nvidia-smi`](https://developer.nvidia.com/nvidia-system-management-interface) to monitor GPU usage on a node where you have a job running. The following command runs `watch` on the node assigned to the given job, which in turn runs `nvidia-smi` every 30 seconds, displaying the output on your terminal.

`$ srun --jobid 123456 --pty watch -n 30 nvidia-smi`

It is possible to launch multiple monitoring commands using [`tmux`](https://docs.alliancecan.ca/Prolonging_terminal_sessions#tmux "tmux"){.wikilink}. The following command launches `htop` and `nvidia-smi` in separate panes to monitor the activity on a node assigned to the given job.

`$ srun --jobid 123456 --pty tmux new-session -d 'htop -u $USER' \; split-window -h 'watch nvidia-smi' \; attach`

Processes launched with `srun` share the resources with the job specified. You should therefore be careful not to launch processes that would use a significant portion of the resources allocated for the job. Using too much memory, for example, might result in the job being killed; using too many CPU cycles will slow down the job.

`<b>`{=html}Noteː`</b>`{=html} The `srun` commands shown above work only to monitor a job submitted with `sbatch`. To monitor an interactive job, create multiple panes with `tmux` and start each process in its own pane.
