---
title: "Job arrays/en"
url: "https://docs.alliancecan.ca/wiki/Job_arrays/en"
category: "General"
last_modified: "2024-02-09T20:27:02Z"
page_id: 7979
display_title: "Job arrays"
---

`<languages />`{=html}

`<i>`{=html}Parent page: [Running jobs](https://docs.alliancecan.ca/Running_jobs "Running jobs"){.wikilink}`</i>`{=html}

If your work consists of a large number of tasks which differ only in some parameter, you can conveniently submit many tasks at once using a `<i>`{=html}job array,`</i>`{=html} also known as a `<i>`{=html}task array`</i>`{=html} or an `<i>`{=html}array job`</i>`{=html}. The individual tasks in the array are distinguished by an environment variable, `$SLURM_ARRAY_TASK_ID`, which Slurm sets to a different value for each task. You set the range of values with the `--array` parameter.

See [Job Array Support](https://slurm.schedmd.com/job_array.html) for more details.

## Examples of the \--array parameter {#examples_of_the___array_parameter}

`sbatch --array=0-7       # $SLURM_ARRAY_TASK_ID takes values from 0 to 7 inclusive`\
`sbatch --array=1,3,5,7   # $SLURM_ARRAY_TASK_ID takes the listed values`\
`sbatch --array=1-7:2     # Step size of 2, same as the previous example`\
`sbatch --array=1-100%10  # Allows no more than 10 of the jobs to run simultaneously`

## A simple example {#a_simple_example}

This job will be scheduled as ten independent tasks. Each task has a separate time limit of 3 hours, and each may start at a different time on a different host.

The script references `$SLURM_ARRAY_TASK_ID` to select an input file (named `<i>`{=html}program_x`</i>`{=html} in our example), or to set a command-line argument for the application (as in `<i>`{=html}program_y`</i>`{=html}).

Using a job array instead of a large number of separate serial jobs has advantages for you and other users. A waiting job array only produces one line of output in squeue, making it easier for you to read its output. The scheduler does not have to analyze job requirements for each task in the array separately, so it can run more efficiently too.

Note that, other than the initial job-submission step with `sbatch`, the load on the scheduler is the same for an array job as for the equivalent number of non-array jobs. The cost of dispatching each array task is the same as dispatching a non-array job. You should not use a job array to submit tasks with very short run times, e.g. much less than an hour. Tasks with run times of only a few minutes should be grouped into longer jobs using [META](https://docs.alliancecan.ca/META:_A_package_for_job_farming "META"){.wikilink}, [GLOST](https://docs.alliancecan.ca/GLOST "GLOST"){.wikilink}, [GNU Parallel](https://docs.alliancecan.ca/GNU_Parallel "GNU Parallel"){.wikilink}, or a shell loop inside a job.

## Example: Multiple directories {#example_multiple_directories}

Suppose you have multiple directories, each with the same structure, and you want to run the same script in each directory. If the directories can be named with sequential numbers then the example above can be easily adapted. If the names are not so systematic, then create a file with the names of the directories, like so:

`$ cat case_list`\
`pacific2016`\
`pacific2017`\
`atlantic2016`\
`atlantic2017`

There are several ways to select a given line from a file; this example uses `sed` to do so:

```{=mediawiki}
{{File
|name=directories_array.sh
|language=bash
|contents=
#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --array=1-4

echo "Starting task $SLURM_ARRAY_TASK_ID"
DIR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" case_list)
cd $DIR

# Place the code to execute here
pwd
ls
}}
```
Cautions:

- Take care that the number of tasks you request matches the number of entries in the file.
- The file `case_list` should not be changed until all the tasks in the array have run, since it will be read each time a new task starts.

## Example: Multiple parameters {#example_multiple_parameters}

Suppose you have a Python script doing certain calculations with some parameters defined in a Python list or a NumPy array such as

The above task can be processed in a job array so that each value of the beta parameter can be treated in parallel. The idea is to pass the `$SLURM_ARRAY_TASK_ID` to the Python script and get the beta parameter based on its value. The Python script becomes

The job submission script is (note the array parameters goes from 0 to 99 like the indexes of the NumPy array)
