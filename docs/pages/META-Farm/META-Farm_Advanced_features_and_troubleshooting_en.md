---
title: "META-Farm: Advanced features and troubleshooting/en"
url: "https://docs.alliancecan.ca/wiki/META-Farm:_Advanced_features_and_troubleshooting/en"
category: "META-Farm"
last_modified: "2025-09-05T17:48:06Z"
page_id: 21476
display_title: "META-Farm: Advanced features and troubleshooting"
---

`<languages />`{=html}

This page presents more advanced features of the [META-Farm](https://docs.alliancecan.ca/META-Farm "META-Farm"){.wikilink} package.

# Resubmitting failed cases automatically {#resubmitting_failed_cases_automatically}

If your farm is particularly large, that is, if it needs more resources than *NJOBS_MAX x job_run_time*, where *NJOBS_MAX* is the maximum number of jobs one is allowed to submit, you will have to run `resubmit.run` after the original farm finishes running\-- perhaps more than once. You can do it by hand, but with META you can also automate this process. To enable this feature, add the `-auto` switch to your `submit.run` or `resubmit.run` command:

`$ submit.run N -auto`

This can be used in either SIMPLE or META mode. If your original `submit.run` command did not have the `-auto` switch, you can add it to `resubmit.run` after the original farm finishes running, to the same effect.

When you add `-auto`, `(re)submit.run` submits one more (serial) job, in addition to the farm jobs. The purpose of this job is to run the `resubmit.run` command automatically right after the current farm finishes running. The job script for this additional job is `resubmit_script.sh`, which should be present in the farm directory; a sample file is automatically copied there when you run `farm_init.run`. The only customization you need to do to this file is to correct the account name in the `#SBATCH -A` line.

If you are using `-auto`, the value of the `NJOBS_MAX` parameter defined in the `config.h` file should be at least one smaller than the largest number of jobs you can submit on the cluster. E.g. if the largest number of jobs one can submit on the cluster is 999 and you intend to use `-auto`, set `NJOBS_MAX` to 998.

When using `-auto`, if at some point the only cases left to be processed are the ones which failed earlier, auto-resubmission will stop, and farm computations will end. This is to avoid an infinite loop on badly-formed cases which will always fail. If this happens, you will have to address the reasons for these cases failing before attempting to resubmit the farm. You can see the relevant messages in the file `farm.log` created in the farm directory.

# Running a post-processing job automatically {#running_a_post_processing_job_automatically}

Another advanced feature is the ability to run a post-processing job automatically once all the cases from table.dat have been **successfully** processed. If any cases failed\-- *i.e.* had a non-zero exit status\-- the post-processing job will not run. To enable this feature, simply create a script for the post-processing job with the name `final.sh` inside the farm directory This job can be of any kind\-- serial, parallel, or an array job.

This feature uses the same script, `resubmit_script.sh`, described for [`-auto`](https://docs.alliancecan.ca/#Resubmitting_failed_cases_automatically "-auto"){.wikilink} above. Make sure `resubmit_script.sh` has the correct account name in the `#SBATCH -A` line.

The automatic post-processing feature also causes more serial jobs to be submitted, above the number you request. Adjust the parameter `NJOBS_MAX` in `config.h` accordingly (*e.g.* if the cluster has a job limit of 999, set it to 998). However, if you use both the auto-resubmit and the auto-post-processing features, they will together only submit *one* additional job. You do not need to subtract 2 from `NJOBS_MAX`.

System messages from the auto-resubmit feature are logged in `farm.log`, in the root farm directory.

# WHOLE_NODE mode {#whole_node_mode}

Starting from the version 1.0.3, meta-farm supports packaging individual serial farming jobs into whole node jobs. This made it possible to use the package on Trillium. This mode is off by default. To enable it, edit the file config.h inside your farm directory. Specifically, you need to set `WHOLE_NODE=1`, and set the variable `NWHOLE` to the number of CPU cores per node (192 for Trillium).

In the WHOLE_NODE mode, the positive integer argument for the `submit.run` command changes its meaning: instead of being the number of meta-jobs, now it is the number of whole nodes to be used in META mode. For example, consider this command:

`$ submit.run 2`

If the WHOLE_NODE mode is enabled, the above command will allocate 2 whole nodes, which will be used to run up to 384 concurrent serial tasks (192 tasks on each node) using META mode (dynamic workload balancing). These tasks are executed as separate threads within whole-node jobs.

The \"-1\" argument for `submit.run` preserves its original meaning: run the farm using the SIMPLE mode. The number of actual (whole node) jobs is computed as `Number_of_cases / NWHOLE`.

Important details:

- The advanced features \"Automatic job resubmission\" and \"Automatic post-processing job\" will only work on Trillium if you place the following line at the end of your \~/.bashrc file:

`module load StdEnv`

- The WHOLE_NODE mode can only be used for serial farming. (That is, it cannot be used for multi-threaded, MPI, or GPU farming).
- The WHOLE_NODE mode can also be used on other clusters (not just on Trillium). It may be advantageous in situations when the queue wait time for whole node jobs becomes shorter that the queue wait time for serial jobs.

# Additional information {#additional_information}

## Using the git repository {#using_the_git_repository}

To use META on a cluster where it is not installed as a module you can clone the package from our git repository:

`$ git clone `[`https://git.computecanada.ca/syam/meta-farm.git`](https://git.computecanada.ca/syam/meta-farm.git)

Then modify your \$PATH variable to point to the `bin` subdirectory of the newly created `meta-farm` directory. Assuming you executed `git clone` inside your home directory, do this:

`$ export PATH=~/meta-farm/bin:$PATH`

Then proceed as shown in the META [Quick start](https://docs.alliancecan.ca/META:_A_package_for_job_farming#Quick_start "Quick start"){.wikilink} from the `farm_init.run` step.

## Passing additional sbatch arguments {#passing_additional_sbatch_arguments}

If you need to use additional `sbatch` arguments (like `--mem 4G, --gres=gpu:1` *etc.*), add them to `job_script.sh` as separate `#SBATCH` lines.

Or if you prefer, you can add them at the end of the `submit.run` or `resubmit.run` command and they will be passed to `sbatch`, *e.g.:*

``` bash
   $  submit.run  -1  --mem 4G
```

## Multi-threaded applications {#multi_threaded_applications}

For [multi-threaded](https://docs.alliancecan.ca/Running_jobs#Threaded_or_OpenMP_job "multi-threaded"){.wikilink} applications (such as those that use [OpenMP](https://docs.alliancecan.ca/OpenMP "OpenMP"){.wikilink}, for example), add the following lines to `job_script.sh`:

``` bash
   #SBATCH --cpus-per-task=N
   #SBATCH --mem=M
   export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
```

\...where *N* is the number of CPU cores to use, and *M* is the total memory to reserve in megabytes. You may also supply `--cpus-per-task=N` and `--mem=M` as arguments to `(re)submit.run`.

## MPI applications {#mpi_applications}

For applications that use [MPI](https://docs.alliancecan.ca/MPI "MPI"){.wikilink}, add the following lines to `job_script.sh`:

``` bash
   #SBATCH --ntasks=N  
   #SBATCH --mem-per-cpu=M
```

\...where *N* is the number of CPU cores to use, and *M* is the memory to reserve for each core, in megabytes. You may also supply `--ntasks=N` and `--mem-per-cpu=M` as arguments to `(re)submit.run`. See [Advanced MPI scheduling](https://docs.alliancecan.ca/Advanced_MPI_scheduling "Advanced MPI scheduling"){.wikilink} for information about more-complicated MPI scenarios.

Also add `srun` before the path to your code inside `single_case.sh`,*e.g.*:

``` bash
   srun  $COMM
```

Alternatively, you can prepend `srun` to each line of `table.dat`:

``` bash
   srun /path/to/mpi_code arg1 arg2
   srun /path/to/mpi_code arg1 arg2
   ...
   srun /path/to/mpi_code arg1 arg2
```

## GPU applications {#gpu_applications}

For applications which use GPUs, modify `job_script.sh` following the guidance at [Using GPUs with Slurm](https://docs.alliancecan.ca/Using_GPUs_with_Slurm "Using GPUs with Slurm"){.wikilink}:

``` bash
#SBATCH --gres=gpu[[:type]:number]
```

You may also wish to copy the utility `~syam/bin/gpu_test` to your `~/bin` directory (only on Nibi), and put the following lines in `job_script.sh` right before the `task.run` line:

``` bash
~/bin/gpu_test
retVal=$?
if [ $retVal -ne 0 ]; then
    echo "No GPU found - exiting..."
    exit 1
fi
```

This will catch those rare situations when there is a problem with the node which renders the GPU unavailable. If that happens to one of your meta-jobs, and you don\'t detect the GPU failure somehow, then the job will try (and fail) to run all your cases from `table.dat`.

## Environment variables and \--export {#environment_variables_and___export}

All the jobs generated by META package inherit the environment present when you run `submit.run` or `resubmit.run`. This includes all the loaded modules and environment variables. META relies on this behaviour for its work, using some environment variables to pass information between scripts. You have to be careful not to break this default behaviour, such as can happen if you use the `--export` switch. If you need to use `--export` in your farm, make sure `ALL` is one of the arguments to this command, *e.g.* `--export=ALL,X=1,Y=2`.

If you need to pass values of custom environment variables to all of your farm jobs (including auto-resubmitted jobs and the post-processing job if there is one), do not use `--export`. Instead, set the variables on the command line as in this example:

``` bash
   $  VAR1=1 VAR2=5 VAR3=3.1416 submit.run ...
```

Here `VAR1, VAR2, VAR3` are custom environment variables which will be passed to all farm jobs.

## Example: Numbered input files {#example_numbered_input_files}

Suppose you have an application called `fcode`, and each case needs to read a separate file from standard input--- say `data.X`, where *X* ranges from 1 to *N_cases*. The input files are all stored in a directory `/home/user/IC`. Ensure `fcode` is on your `$PATH` (*e.g.*, put `fcode` in `~/bin`, and ensure `/home/$USER/bin` is added to `$PATH` in `~/.bashrc`), or use a full path to `fcode` in `table.dat`. Create `table.dat` in the farm META directory like this:

`  fcode < /home/user/IC/data.1`\
`  fcode < /home/user/IC/data.2`\
`  fcode < /home/user/IC/data.3`\
`  ...`

You might wish to use a shell loop to create `table.dat`, *e.g.*:

``` bash
   $  for ((i=1; i<=100; i++)); do echo "fcode < /home/user/IC/data.$i"; done >table.dat
```

## Example: Input file must have the same name {#example_input_file_must_have_the_same_name}

Some applications expect to read input from a file with a prescribed and unchangeable name, like `INPUT` for example. To handle this situation each case must run in its own subdirectory, and you must create an input file with the prescribed name in each subdirectory. Suppose for this example that you have prepared the different input files for each case and stored them in `/path/to/data.X`, where *X* ranges from 1 to *N_cases*. Your `table.dat` can contain nothing but the application name, over and over again:

`  /path/to/code`\
`  /path/to/code`\
`  ...`

Add a line to `single_case.sh` which copies the input file into the farm *sub*directory for each case\-- the first line in the example below:

``` bash
   cp /path/to/data.$ID INPUT
   $COMM
   STATUS=$?
```

## Using all the columns in the cases table explicitly {#using_all_the_columns_in_the_cases_table_explicitly}

The examples shown so far assume that each line in the cases table is an executable statement, starting with either the name of the executable file (when it is on your `$PATH`) or the full path to the executable file, and then listing the command line arguments particular to that case, or something like `< input.$ID` if your code expects to read a standard input file.

In the most general case, you may want to be able to access all the columns in the table individually. That can be done by modifying `single_case.sh`:

``` bash
...
# ++++++++++++  This part can be customized:  ++++++++++++++++
#  $ID contains the case id from the original table
#  $COMM is the line corresponding to the case $ID in the original table, without the ID field
mkdir RUN$ID
cd RUN$ID

# Converting $COMM to an array:
COMM=( $COMM )
# Number of columns in COMM:
Ncol=${#COMM[@]}
# Now one can access the columns individually, as ${COMM[i]} , where i=0...$Ncol-1
# A range of columns can be accessed as ${COMM[@]:i:n} , where i is the first column
# to display, and n is the number of columns to display
# Use the ${COMM[@]:i} syntax to display all the columns starting from the i-th column
# (use for codes with a variable number of command line arguments).

# Call the user code here.
...

# Exit status of the code:
STATUS=$?
cd ..
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
...
```

For example, you might need to provide to your code *both* a standard input file *and* a variable number of command line arguments. Your cases table will look like this:

`  /path/to/IC.1 0.1`\
`  /path/to/IC.2 0.2 10`\
`  ...`

The way to implement this in `single_case.sh` is as follows:

``` bash
# Call the user code here.
/path/to/code ${COMM[@]:1} < ${COMM[0]}
```

## Reducing waste {#reducing_waste}

Here is one potential problem when one is running multiple cases per job: What if the number of running meta-jobs times the requested run-time per meta-job (say, 3 days) is not enough to process all your cases? E.g., you managed to start the maximum allowed 1000 meta-jobs, each of which has a 3-day run-time limit. That means that your farm can only process all the cases in a single run if the *average_case_run_time x N_cases \< 1000 x 3d = 3000* CPU days. Once your meta-jobs start hitting the 3-day run-time limit, they will start dying in the middle of processing one of your cases. This will result in up to 1000 interrupted cases calculations. This is not a big deal in terms of completing the work\-\-- `resubmit.run` will find all the cases which failed or never ran, and will restart them automatically. But this can become a waste of CPU cycles. On average, you will be wasting *0.5 x N_jobs x average_case_run_time*. E.g., if your cases have an average run-time of 1 hour, and you have 1000 meta-jobs running, you will waste about 500 CPU-hours or about 20 CPU-days, which is not acceptable.

Fortunately, the scripts we are providing have some built-in intelligence to mitigate this problem. This is implemented in `task.run` as follows:

- The script measures the run-time of each case, and adds the value as one line in a scratch file `times` created in directory `/home/$USER/tmp/$NODE.$PID/`. (See [Output files](https://docs.alliancecan.ca/#Output_files "Output files"){.wikilink}.) This is done by all running meta-jobs.
- Once the first 8 cases were computed, one of the meta-jobs will read the contents of the file `times` and compute the largest 12.5% quantile for the current distribution of case run-times. This will serve as a conservative estimate of the run-time for your individual cases, *dt_cutoff*. The current estimate is stored in file `dt_cutoff` in `/home/$USER/tmp/$NODE.$PID/`.
- From now on, each meta-job will estimate if it has the time to finish the case it is about to start computing, by ensuring that *t_finish - t_now \> dt_cutoff*. Here, *t_finish* is the time when the job will die because of the job\'s run-time limit, and *t_now* is the current time. If it computes that it doesn\'t have the time, it will exit early, which will minimize the chance of a case aborting half-way due to the job\'s run-time limit.
- At every subsequent power of two number of computed cases (8, then 16, then 32 and so on) *dt_cutoff* is recomputed using the above algorithm. This will make the *dt_cutoff* estimate more and more accurate. Power of two is used to minimize the overheads related to computing *dt_cutoff*; the algorithm will be equally efficient for both very small (tens) and very large (many thousands) number of cases.
- The above algorithm reduces the amount of CPU cycles wasted due to jobs hitting the run-time limit by a factor of 8, on average.

As a useful side effect, every time you run a farm you get individual run-times for all of your cases stored in `/home/$USER/tmp/$NODE.$PID/times`. You can analyze that file to fine-tune your farm setup, for profiling your code, etc.

# Troubleshooting

Here we explain typical error messages you might get when using this package.

## Problems affecting multiple commands {#problems_affecting_multiple_commands}

### \"Non-farm directory, or no farm has been submitted; exiting\" {#non_farm_directory_or_no_farm_has_been_submitted_exiting}

Either the current directory is not a farm directory, or you never ran `submit.run` for this farm.

## Problems with submit.run {#problems_with_submit.run}

### Wrong first argument: XXX (should be a positive integer or -1) ; exiting {#wrong_first_argument_xxx_should_be_a_positive_integer_or__1_exiting}

Use the correct first argument: -1 for the SIMPLE mode, or a positive integer N (number of requested meta-jobs) for the META mode.

### \"lockfile is not on path; exiting\" {#lockfile_is_not_on_path_exiting}

Make sure the utility `lockfile` is on your \$PATH. This utility is critical for this package. It provides serialized access of meta-jobs to the `table.dat` file, that is, it ensures that two different meta-jobs do not read the same line of `table.dat` at the same time.

### \"Non-farm directory (config.h, job_script.sh, single_case.sh, and/or table.dat are missing); exiting\" {#non_farm_directory_config.h_job_script.sh_single_case.sh_andor_table.dat_are_missing_exiting}

Either the current directory is not a farm directory, or some important files are missing. Change to the correct (farm) directory, or create the missing files.

### \"-auto option requires resubmit_script.sh file in the root farm directory; exiting\" {#auto_option_requires_resubmit_script.sh_file_in_the_root_farm_directory_exiting}

You used the `-auto` option, but you forgot to create the `resubmit_script.sh` file inside the root farm directory. A sample `resubmit_script.sh` is created automatically when you use `farm_init.run`.

### \"File table.dat doesn\'t exist. Exiting\" {#file_table.dat_doesnt_exist._exiting}

You forgot to create the `table.dat` file in the current directory, or perhaps you are running `submit.run` not inside one of your farm sub-directories.

### \"Job runtime sbatch argument (-t or \--time) is missing in job_script.sh. Exiting\" {#job_runtime_sbatch_argument__t_or___time_is_missing_in_job_script.sh._exiting}

Make sure you provide a run-time limit for all meta-jobs as an `#SBATCH` argument inside your `job_script.sh` file. The run-time is the only one which cannot be passed as an optional argument to `submit.run`.

### \"Wrong job runtime in job_script.sh - nnn . Exiting\" {#wrong_job_runtime_in_job_script.sh___nnn_._exiting}

You didn\'t format properly the run-time argument inside your `job_script.sh` file.

===\"Something wrong with sbatch farm submission; jobid=XXX; aborting\"=== ===\"Something wrong with a auto-resubmit job submission; jobid=XXX; aborting\"=== With either of the two messages, there was an issue with submitting jobs with `sbatch`. The cluster\'s scheduler might be misbehaving, or simply too busy. Try again a bit later.

### \"Couldn\'t create subdirectories inside the farm directory ; exiting\" {#couldnt_create_subdirectories_inside_the_farm_directory_exiting}

### \"Couldn\'t create the temp directory XXX ; exiting\" {#couldnt_create_the_temp_directory_xxx_exiting}

### \"Couldn\'t create a file inside XXX ; exiting\" {#couldnt_create_a_file_inside_xxx_exiting}

With any of these three messages, something is wrong with a file system: Either permissions got messed up, or you have exhausted a quota. Fix the issue(s), then try again.

## Problems with resubmit.run {#problems_with_resubmit.run}

### \"Jobs are still running/queued; cannot resubmit\" {#jobs_are_still_runningqueued_cannot_resubmit}

You cannot use `resubmit.run` until all meta-jobs from this farm have finished running. Use `list.run` or `queue.run` to check the status of the farm.

### \"No failed/unfinished jobs; nothing to resubmit\" {#no_failedunfinished_jobs_nothing_to_resubmit}

Your farm was 100% processed. There are no more (failed or never-ran) cases to compute.

## Problems with running jobs {#problems_with_running_jobs}

### \"Too many failed (very short) cases - exiting\" {#too_many_failed_very_short_cases___exiting}

This happens if the first `$N_failed_max` cases are very short\-- less than `$dt_failed` seconds in duration. See the discussion in section [job_script.sh](https://docs.alliancecan.ca/#job_script.sh "job_script.sh"){.wikilink} above. Determine what is causing the cases to fail and fix that, or else adjust the `$N_failed_max` and `$dt_failed` values in `config.h`.

### \"lockfile is not on path on node XXX\" {#lockfile_is_not_on_path_on_node_xxx}

As the error message suggests, somehow the utility `lockfile` is not on your `$PATH` on some node. Use `which lockfile` to ensure that the utility is somewhere in your `$PATH`. If it is in your `$PATH` on a login node, then something went wrong on that particular compute node, for example a file system may have failed to mount.

### \"Exiting after processing one case (-1 option)\" {#exiting_after_processing_one_case__1_option}

This is not an error message. It simply tells you that you submitted the farm with `submit.run -1` (one case per job mode), so each meta-job is exiting after processing a single case.

### \"Not enough runtime left; exiting.\" {#not_enough_runtime_left_exiting.}

This message tells you that the meta-job would likely not have enough time left to process the next case (based on the analysis of run-times for all the cases processed so far), so it is exiting early.

### \"No cases left; exiting.\" {#no_cases_left_exiting.}

This is not an error message. This is how each meta-job normally finishes, when all cases have been computed.

### \"Only failed cases left; cannot auto-resubmit; exiting\" {#only_failed_cases_left_cannot_auto_resubmit_exiting}

This can only happen if you used the `-auto` switch when submitting the farm. Find the failed cases with `Status.run -f`, fix the issue(s) causing the cases to fail, then run `resubmit.run`.

*Parent page:* [META: A package for job farming](https://docs.alliancecan.ca/META:_A_package_for_job_farming "META: A package for job farming"){.wikilink}
