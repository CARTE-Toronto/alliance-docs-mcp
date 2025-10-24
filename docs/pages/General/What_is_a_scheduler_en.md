---
title: "What is a scheduler?/en"
url: "https://docs.alliancecan.ca/wiki/What_is_a_scheduler%3F/en"
category: "General"
last_modified: "2023-07-20T21:54:02Z"
page_id: 3598
display_title: "What is a scheduler?"
---

`<languages />`{=html}

## What\'s a job? {#whats_a_job}

On computers, we are most often familiar with graphical user interfaces (GUIs). There are windows, menus, buttons; we click here and there and the system responds. On our servers, the environment is different. To begin with, you control it by typing, not clicking. This is called a [command line interface](https://docs.alliancecan.ca/Linux_introduction "command line interface"){.wikilink}. Furthermore, a program you would like to run may not begin immediately, but may instead be put on a waiting list. When the necessary CPU cores are available it will begin, otherwise jobs would interfere with each other leading to performance loss.

You prepare a small text file called a `<i>`{=html}job script`</i>`{=html} that basically says what program to run, where to get the input, and where to put the output. You `<i>`{=html}submit`</i>`{=html} this job script to a piece of software called the `<i>`{=html}scheduler`</i>`{=html} which decides when and where it will run. Once the job has finished, you can retrieve the results of the calculation. Normally there is no interaction between you and the program while the job is running, although you can check on its progress if you wish.

Here\'s a very simple job script:

It runs the programs `echo` and `sleep`, there is no input, and the output will go to a default location. Lines starting with `#SBATCH` are directives to the scheduler, providing information about what the job needs to run. This job, for example, only needs one minute of run time (00:01:00).

## The job scheduler {#the_job_scheduler}

The job scheduler is a piece of software with multiple responsibilities. It must

- maintain a database of jobs,
- enforce policies regarding limits and priorities,
- ensure resources are not overloaded, for example by only assigning each CPU core to one job at a time,
- decide which jobs to run and on which compute nodes,
- launch them on those nodes, and
- clean up after each job finishes.

On our clusters, these responsibilities are handled by the [Slurm Workload Manager](https://en.wikipedia.org/wiki/Slurm_Workload_Manager). All the examples and syntax shown on this page are for Slurm.

## Requesting resources {#requesting_resources}

You use the job script to ask for the resources needed to run your calculation. Among the resources associated with a job are `<i>`{=html}time`</i>`{=html} and `<i>`{=html}number of processors`</i>`{=html}. In the example above, the time requested is one minute and there will be one processor allocated by default since no specific number is given. Please refer to [Examples of job scripts](https://docs.alliancecan.ca/Running_jobs#Examples_of_job_scripts "Examples of job scripts"){.wikilink} for other types of requests such as multiple processors, memory capacity and special processors such as [GPUs](https://en.wikipedia.org/wiki/General-purpose_computing_on_graphics_processing_units).

It is important to specify those parameters well. If you ask for less than the calculation needs, the job will be killed for exceeding the requested time or memory limit. If you ask for more than it needs, the job may wait longer than necessary before it starts, and once running it will needlessly prevent others from using those resources.

## A basic Slurm job {#a_basic_slurm_job}

We can submit the job script `simple_job.sh` shown above with [sbatch](https://slurm.schedmd.com/sbatch.html):

``` bash
[someuser@host ~]$ sbatch simple_job.sh
Submitted batch job 1234
[someuser@host ~]$ sq
   JOBID     USER      ACCOUNT      NAME  ST  TIME_LEFT NODES CPUS    GRES MIN_MEM NODELIST (REASON)
    1234 someuser def-someprof  simple_j   R       0:33     1    1  (null)    256M blg9876 (None)
[someuser@host ~]$ cat slurm-1234.out
Hello, world!
```

Look at the ST column in the output of [ sq](https://docs.alliancecan.ca/Running_jobs#Monitoring_jobs " sq"){.wikilink} to determine the status of your jobs. The two most common states are PD for `<i>`{=html}pending`</i>`{=html} and R for `<i>`{=html}running`</i>`{=html}. When the job has finished, it no longer appears in the `sq` output.

Notice that each job is assigned a `<i>`{=html}job ID`</i>`{=html}, a unique identification number printed when you submit the job \-\-- 1234 in this example. You can have more than one job in the system at a time, and the ID number can be used to distinguish them even if they have the same name. And finally, because we didn\'t specify anywhere else to put it the output is placed in a file named with the same job ID number, `slurm‑1234.out`.

You can also specify options to `sbatch` on the command line. So for example,

`[someuser@host ~]$ sbatch --time=00:30:00 simple_job.sh `

will change the time limit of the job to 30 minutes. Any option can be overridden in this way.

## Choosing where the output goes {#choosing_where_the_output_goes}

If you want the output file to have a more distinctive name than `slurm‑1234.out`, you can use `--output` to change it. The following script sets a `<i>`{=html}job name`</i>`{=html} which will appear in the `squeue` output, and sends the output to a file prefixed with the job name and containing the job ID number, for exemple `<i>`{=html}test-1234.out`</i>`{=html}.

Error output will normally appear in the same file, just as it would if you were typing commands interactively. If you wish you can split the standard error channel (stderr) from the standard output channel (stdout) by specifying a file name with the `‑e` option.

## Accounts and projects {#accounts_and_projects}

Information about your job, like how long it waited, how long it ran, and how many cores it used, is recorded so we can monitor our quality of service and so we can report to our funders how their money is spent. Every job must have an associated `<i>`{=html}account name`</i>`{=html} corresponding to a [ resource allocation project](https://docs.alliancecan.ca/Frequently_Asked_Questions_about_the_CCDB#Resource_Allocation_Projects_(RAP) " resource allocation project"){.wikilink}.

`#SBATCH --account=def-user-ab`

If you try to submit a job with `sbatch` without supplying an account name, and one is needed, you will be shown a list of valid account names to choose from.
