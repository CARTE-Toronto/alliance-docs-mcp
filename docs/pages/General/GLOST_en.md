---
title: "GLOST/en"
url: "https://docs.alliancecan.ca/wiki/GLOST/en"
category: "General"
last_modified: "2025-03-05T21:05:05Z"
page_id: 7759
display_title: "GLOST"
---

`<languages />`{=html}

# Introduction

[GLOST](https://github.com/cea-hpc/glost), the Greedy Launcher Of Small Tasks, is a tool for running many sequential jobs of short or variable duration, or for doing parameter sweeps. It works like [GNU parallel](https://docs.alliancecan.ca/GNU_Parallel "GNU parallel"){.wikilink} or [job arrays](https://docs.alliancecan.ca/Job_arrays "job arrays"){.wikilink} but with a simpler syntax. GLOST uses a wrapper called `glost_launch` and [MPI](https://docs.alliancecan.ca/MPI "MPI"){.wikilink} commands `srun`, `mpiexec` and `mpirun`. Jobs are grouped into one text file, `<b>`{=html}list_glost_tasks.txt`</b>`{=html}, which is used as an argument for `glost_launch`.

`<b>`{=html}GLOST`</b>`{=html} can be used in the following situations:

- large number of serial jobs with comparative runtime,
- large number of short serial jobs,
- serial jobs with different parameters (parameter sweep).

The idea behind using GLOST consists on bundling serial jobs and run them as an MPI job. It can use multiple cores (one or more nodes). This will reduce considerably the number of the jobs on the queue, and therefore, reduce the stress on the [ scheduler](https://docs.alliancecan.ca/Running_jobs " scheduler"){.wikilink}.

As an alternative, you may also want to consider the [ META](https://docs.alliancecan.ca/META:_A_package_for_job_farming " META"){.wikilink} software package developed by our staff, which has some important advantages over GLOST. In particular, with META the queue wait time may be significantly shorter than with GLOST, and META overheads are smaller (fewer wasted CPU cycles). In addition, META has a convenient mechanism for re-submitting all the computations that never ran or failed. Finally, unlike GLOST, META can be used for all kinds of jobs; serial, multi-threaded, MPI, GPU, or hybrid.

`<b>`{=html}Note:`</b>`{=html} please read this document until the end and if you think that your workflow can fit within this framework, contact [Technical support](https://docs.alliancecan.ca/Technical_support "Technical support"){.wikilink} to help you change your workflow.

# Advantage of using GLOST {#advantage_of_using_glost}

GLOST is used to bundle a set of serial jobs into one single or more MPI jobs depending on the duration of the jobs and their number.

Submitting a large number of serial jobs at once can slow down the scheduler leading in most cases to a slow response and frequent time out from `sbatch` or `squeue` requests. The idea is to put all the serial tasks into one single file, for example `<b>`{=html}list_glost_tasks.txt`</b>`{=html}, and submit an MPI job using the `glost_launch` wrapper. This will reduce considerably the number of the jobs on the queue leading to less requests to the scheduler compared to the situation if the jobs are submitted separately. Using GLOST to submit serial jobs reduces the stress experienced by the Slurm scheduler when a large number of jobs are submitted at the same time without any delay.

Using GLOST, the user will submit and handle few MPI jobs rather than hundreds or thousands serial jobs.

# Modules

GLOST uses OpenMPI to run a set of serial tasks as an MPI job. For each OpenMPI version, a corresponding module of Glost is installed. To use it, make sure to load OpenMPI and Glost modules. For more information, please refer to the page [using modules](https://docs.alliancecan.ca/using_modules "using modules"){.wikilink}. To see the current installed modules on our systems, use `module spider glost`. Before submitting a job, make sure that you can load GLOST along with the other modules that are required to run your application.

``` bash
$  module spider glost/0.3.1

--------------------------------------------------------------------------------------------------------------------------------------
  glost: glost/0.3.1
--------------------------------------------------------------------------------------------------------------------------------------
    Description:
      This is GLOST, the Greedy Launcher Of Small Tasks. 

    Properties:
      Tools for development / Outils de développement

    You will need to load all module(s) on any one of the lines below before the "glost/0.3.1" module is available to load.

      StdEnv/2023  gcc/12.3  openmpi/4.1.5
      StdEnv/2023  intel/2023.2.1  openmpi/4.1.5
 
    Help:
      
      Description
      ===========
      This is GLOST, the Greedy Launcher Of Small Tasks.
      
      
      More information
      ================
       - Homepage: https://github.com/cea-hpc/glost
```

If there is already an OpenMPI module in your environment, like the default environment, adding `module load glost` to your list of the modules needed for your application, is sufficient to activate GLOST. Use `module list` to make sure that GLOST module is loaded along with other modules before submitting your job.

# How to use GLOST? {#how_to_use_glost}

## GLOST syntax {#glost_syntax}

The general syntax of GLOST can take one of the following forms:

``` bash
srun glost_launch list_glost_tasks.txt

mpiexec glost_launch list_glost_tasks.txt 

mpirun glost_launch list_glost_tasks.txt
```

## Number of cores versus number of jobs {#number_of_cores_versus_number_of_jobs}

GLOST uses a cyclic distribution to distribute the serial jobs among the available cores for the job. The GLOST wrapper picks the first lines from the list of jobs and assign one processor to each job (or line from the list) and when one or more processors are done with the first tasks, GLOST will assign them the following lines on the list until the end of the list or until the job runs out of time. Therefore, the number of cores may not necessarily match the number of requested jobs in the list. However, in order to optimize the use of resources, one may need to make sure that the serial jobs have similar runtime and they can be distributed evenly among the cores asked for. Different situations can be treated:

- If you have a large number of very short serial jobs to run (hundreds or thousands of jobs with a very short time, few minutes for example), you submit one or more GLOST jobs that will run a set of serial jobs using few cores. The jobs can be scheduled for short time and by node to take advantage of the back-filling and the scheduler.
- If you have tens to hundreds of relatively short runtime jobs (an hour or so), you can bundle them into one or more GLOST jobs.
- If you have many long serial jobs with similar runtimes, they can also be used as a GLOST job.

## Estimation of the wall time for GLOST job {#estimation_of_the_wall_time_for_glost_job}

Before running a GLOST job, try to estimate the runtime for your serial jobs. It can be used to estimate the wall time for your GLOST job.

Let us suppose you want to run a GLOST job where you have a list of `<b>`{=html}Njobs`</b>`{=html} of similar jobs where each job take `<b>`{=html}t0`</b>`{=html} as a runtime using 1 processor. The total runtime for all these jobs will be: `<b>`{=html}t0\*Njobs`</b>`{=html}

Now, if you are going to use `<b>`{=html}Ncores`</b>`{=html} to run your GLOST job, the time required for this job will be: `<b>`{=html}wt = t0\*Njobs/Ncores`</b>`{=html}.

`<b>`{=html}Note:`</b>`{=html} An MPI job is often designed so that MPI processes need to exchange information. Designs like this can spend a large fraction of time on communication, and so wind up doing less computation. Many, small, dependent communications can reduce the efficiency of the code. In contrast, GLOST uses MPI but only to start entirely serial jobs, which means that communication overhead is relatively infrequent. You could write the same program yourself, using MPI directly, but GLOST provides nearly the same efficiency, without the effort of writing MPI.

## Choosing the memory {#choosing_the_memory}

GLOST uses MPI to run serial jobs and the memory per core should be the same as the memory required for the serial job if it runs separately. Use `--mem-per-cpu` instead of `--mem` in your Slurm script.

## Create the list of tasks {#create_the_list_of_tasks}

Before submitting a job using GLOST, create a text file,`<b>`{=html}list_glost_tasks.txt`</b>`{=html}, that contains all the commands needed to run the serial jobs: one job per line. Ideally, one has to choose jobs with similar runtime in order to optimize the use of resources asked for. The GLOST job can run all the tasks in one or multiple directories. If you run all the jobs in one directory, make sure that the output from the different jobs do not overlap or use the same temporary or output files. To do so, standard output may be redirected to a file with a variable indicating the argument or the option used to run the corresponding jobs. In case of the jobs use similar temporary or output files, you may need to create a directory for each task: one directory for each argument or option that correspond to a particular job.

`<b>`{=html}Note:`</b>`{=html} one job may contain one command or multiple commands executed one after another. The commands should be separated by `&&`.

Here is an example of the file `<b>`{=html}list_glost_example.txt`</b>`{=html} with 8 jobs: `<tabs>`{=html} `<tab name="Script">`{=html}

`</tab>`{=html}

`<tab name="List of tasks">`{=html}

`</tab>`{=html}

`</tabs>`{=html} `<b>`{=html}Note:`</b>`{=html} the above example cannot be executed. The commands are not defined. It shows only:

- a simple syntax for a list of jobs, `<b>`{=html}list_glost_tasks.txt`</b>`{=html} that will serve as an argument for the `glost_launch` wrapper;
- a typical script to submit the job.

Both the list of jobs and the script should be adapted to your workflow.

## List of jobs to run in one directory {#list_of_jobs_to_run_in_one_directory}

GLOST can be used to run a set or a list of serial jobs in one directory. To avoid the overlap of the results, one has to make sure that the different jobs will not use the same temporary or output file. This can be achieved by adding arguments to differentiate the different jobs. In the following example, we have a list of 10 tasks. Each task may contain one or more commands. In this example, each job runs three commands one after another:

- `<b>`{=html}First command:`</b>`{=html} Fix a variable `<b>`{=html}nargument`</b>`{=html}. This could be a parameter or a variable to pass to the program for example.
- `<b>`{=html}Second command:`</b>`{=html} run the program. For testing, we have used the command `sleep 360`. This should be replaced by the command line to run your application. For example: `./my_first_prog < first_input_file.txt > first_output_file.txt`
- `<b>`{=html}Third command:`</b>`{=html} If needed, add one or more commands that will be executed just after the previous ones. All the commands should be separated by `&&`. For testing, we have used the command: `` echo ${nargument}.`hostname` > log_${nargument}.txt ``. For this command, we print out the argument and the `hostname` to a file log\_\${nargument}.txt. Similarly to the second command, this line should be replaced by another command line to run an application just after the previous one if needed. For example: `./my_second_prog < second_input_file.txt > second_output_file.txt`.

`<tabs>`{=html} `<tab name="Script">`{=html}

`</tab>`{=html}

`<tab name="List of tasks">`{=html} ``{{File
  |name=list_glost_tasks.txt
  |lang="txt"
  |contents=
nargument=20 && sleep 360 && echo ${nargument}.`hostname` > log_${nargument}.txt
nargument=21 && sleep 360 && echo ${nargument}.`hostname` > log_${nargument}.txt
nargument=22 && sleep 360 && echo ${nargument}.`hostname` > log_${nargument}.txt
nargument=23 && sleep 360 && echo ${nargument}.`hostname` > log_${nargument}.txt
nargument=24 && sleep 360 && echo ${nargument}.`hostname` > log_${nargument}.txt
nargument=25 && sleep 360 && echo ${nargument}.`hostname` > log_${nargument}.txt
nargument=26 && sleep 360 && echo ${nargument}.`hostname` > log_${nargument}.txt
nargument=27 && sleep 360 && echo ${nargument}.`hostname` > log_${nargument}.txt
nargument=28 && sleep 360 && echo ${nargument}.`hostname` > log_${nargument}.txt
nargument=29 && sleep 360 && echo ${nargument}.`hostname` > log_${nargument}.txt
}}``{=mediawiki} `</tab>`{=html}

`</tabs>`{=html}

`<b>`{=html}Note:`</b>`{=html} In the above example, we have used 2 cores and a list of 10 jobs. GLOST will assign the first two jobs (two first lines) to the available processors, and whenever one and/or both of them are done with the first set of jobs, they will continue with the following jobs until the end of the list.

## List of jobs to run in separate directories {#list_of_jobs_to_run_in_separate_directories}

Similarly to the previous case, GLOST can be used to run multiple serial jobs where each one is executed in a dedicated directory. This could be useful to run a program that uses files (temporary, input and/or output) with the same names in order to avoid the crash of the jobs or an overlap of the results from the different jobs. To do so, one has to make sure to create the input files and a directory for each job before running GLOST. It can be also achieved if included within the line commands as shown in the following example: `<tabs>`{=html} `<tab name="Script">`{=html}

`</tab>`{=html}

`<tab name="List of tasks">`{=html} ``{{File
  |name=list_glost_tasks.txt
  |lang="txt"
  |contents=
nargument=20 && mkdir -p RUN_${nargument} && cd RUN_${nargument} && sleep 360 && echo ${nargument}.`hostname` > log_run.txt
nargument=21 && mkdir -p RUN_${nargument} && cd RUN_${nargument} && sleep 360 && echo ${nargument}.`hostname` > log_run.txt
nargument=22 && mkdir -p RUN_${nargument} && cd RUN_${nargument} && sleep 360 && echo ${nargument}.`hostname` > log_run.txt
nargument=23 && mkdir -p RUN_${nargument} && cd RUN_${nargument} && sleep 360 && echo ${nargument}.`hostname` > log_run.txt
nargument=24 && mkdir -p RUN_${nargument} && cd RUN_${nargument} && sleep 360 && echo ${nargument}.`hostname` > log_run.txt
nargument=25 && mkdir -p RUN_${nargument} && cd RUN_${nargument} && sleep 360 && echo ${nargument}.`hostname` > log_run.txt
nargument=26 && mkdir -p RUN_${nargument} && cd RUN_${nargument} && sleep 360 && echo ${nargument}.`hostname` > log_run.txt
nargument=27 && mkdir -p RUN_${nargument} && cd RUN_${nargument} && sleep 360 && echo ${nargument}.`hostname` > log_run.txt
nargument=28 && mkdir -p RUN_${nargument} && cd RUN_${nargument} && sleep 360 && echo ${nargument}.`hostname` > log_run.txt
nargument=29 && mkdir -p RUN_${nargument} && cd RUN_${nargument} && sleep 360 && echo ${nargument}.`hostname` > log_run.txt
nargument=30 && mkdir -p RUN_${nargument} && cd RUN_${nargument} && sleep 360 && echo ${nargument}.`hostname` > log_run.txt
nargument=31 && mkdir -p RUN_${nargument} && cd RUN_${nargument} && sleep 360 && echo ${nargument}.`hostname` > log_run.txt
}}``{=mediawiki} `</tab>`{=html}

`</tabs>`{=html}

## Restarting a GLOST job {#restarting_a_glost_job}

If you underestimated the wall time for your GLOST job, it may require to be restarted to complete the list of the jobs that were inserted in the list of glost tasks. In this case, make sure to identify the jobs that are already done in order to not run them again. Once identified, remove the corresponding lines from the list of the tasks or create a new list of the jobs that contain the remaining jobs from the previous GLOST job and resubmit your script using the new list as an argument for the `glost_launch` wrapper.

## More examples {#more_examples}

If you are an advanced user and familiar with scripting, you may have a look at the examples by making a copy of the original scripts and adapting them to your workflow.

After loading GLOST module, the examples can be copied to your local directory by running the command:

``` bash
cp -r $EBROOTGLOST/examples Glost_Examples
```

The copy of the examples will be saved under the directory: Glost_Examples

# Related links {#related_links}

- [META-Farm](https://docs.alliancecan.ca/META-Farm "META-Farm"){.wikilink}
- [GNU Parallel](https://docs.alliancecan.ca/GNU_Parallel "GNU Parallel"){.wikilink}
- [Job arrays](https://docs.alliancecan.ca/Job_arrays "Job arrays"){.wikilink}
- [MPI jobs](https://docs.alliancecan.ca/MPI "MPI jobs"){.wikilink}
- [Running jobs](https://docs.alliancecan.ca/Running_jobs "Running jobs"){.wikilink}
