---
title: "Tutoriel Apprentissage machine/en"
url: "https://docs.alliancecan.ca/wiki/Tutoriel_Apprentissage_machine/en"
category: "General"
last_modified: "2023-04-03T19:08:59Z"
page_id: 12411
display_title: "Machine Learning tutorial"
---

`<languages />`{=html}

This page is a beginner\'s manual concerning how to port a machine learning job to one of our clusters.

## Step 1: Remove all graphical display {#step_1_remove_all_graphical_display}

Edit your program such that it doesn\'t use a graphical display. All graphical results will have to be written on disk, and visualized on your personal computer, when the job is finished. For example, if you show plots using matplotlib, you need to [write the plots to image files instead of showing them on screen](https://stackoverflow.com/questions/4706451/how-to-save-a-figure-remotely-with-pylab).

## Step 2: Archiving a data set {#step_2_archiving_a_data_set}

Shared storage on our clusters is not designed to handle lots of small files (they are optimized for very large files). Make sure that the data set which you need for your training is an archive format like `tar`, which you can then transfer to your job\'s compute node when the job starts. `<b>`{=html}If you do not respect these rules, you risk causing enormous numbers of I/O operations on the shared filesystem, leading to performance issues on the cluster for all of its users.`</b>`{=html} If you want to learn more about how to handle collections of large number of files, we recommend that you spend some time reading [this page](https://docs.alliancecan.ca/Handling_large_collections_of_files "this page"){.wikilink}.

Assuming that the files which you need are in the directory `mydataset`:

`$ tar cf mydataset.tar mydataset/*`

The above command does not compress the data. If you believe that this is appropriate, you can use `tar czf`.

## Step 3: Preparing your virtual environment {#step_3_preparing_your_virtual_environment}

[Create a virtual environment](https://docs.alliancecan.ca/Python#Creating_and_using_a_virtual_environment "Create a virtual environment"){.wikilink} in your home space.

For details on installation and usage of machine learning frameworks, refer to our documentation:

- [PyTorch](https://docs.alliancecan.ca/PyTorch "PyTorch"){.wikilink}
- [TensorFlow](https://docs.alliancecan.ca/TensorFlow "TensorFlow"){.wikilink}

## Step 4: Interactive job (salloc) {#step_4_interactive_job_salloc}

We recommend that you try running your job in an [interactive job](https://docs.alliancecan.ca/Running_jobs#Interactive_jobs "interactive job"){.wikilink} before submitting it using a script (discussed in the following section). You can diagnose problems more quickly using an interactive job. An example of the command for submitting such a job is:

`$ salloc --account=def-someuser --gres=gpu:1 --cpus-per-task=3 --mem=32000M --time=1:00:00`

Once the job has started:

- Activate your virtual environment.
- Try to run your program.
- Install any missing modules if necessary. Since the compute nodes don\'t have internet access, you will have to install them from a login node. Please refer to our documentation on [virtual environments](https://docs.alliancecan.ca/Python#Creating_and_using_a_virtual_environment "virtual environments"){.wikilink}.
- Note the steps that you took to make your program work.

**Now is a good time to verify that your job reads and writes as much as possible on the compute node\'s local storage (`$SLURM_TMPDIR`) and as little as possible on the [shared filesystems (home, scratch and project)](https://docs.alliancecan.ca/Storage_and_file_management "shared filesystems (home, scratch and project)"){.wikilink}.**

## Step 5: Scripted job (sbatch) {#step_5_scripted_job_sbatch}

You must [submit your jobs](https://docs.alliancecan.ca/Running_jobs#Use_sbatch_to_submit_jobs "submit your jobs"){.wikilink} using a script in conjunction with the `sbatch` command, so that they can be entirely automated as a batch process. Interactive jobs are just for preparing and debugging your jobs, so that you can execute them fully and/or at scale using `sbatch`.

### Important elements of a `sbatch` script {#important_elements_of_a_sbatch_script}

1.  Account that will be \"billed\" for the resources used
2.  Resources required:
    1.  Number of CPUs, suggestion: 6
    2.  Number of GPUs, suggestion: 1 (**Use one (1) single GPU, unless you are certain that your program can use several. By default, TensorFlow and PyTorch use just one GPU.**)
    3.  Amount of memory, suggestion: `32000M`
    4.  Duration (Maximum Béluga: 7 days, Graham and Cedar: 28 days)
3.  *Bash* commands:
    1.  Preparing your environment (modules, virtualenv)
    2.  Transferring data to the compute node
    3.  Starting the executable

### Example script {#example_script}

### Checkpointing a long-running job {#checkpointing_a_long_running_job}

We recommend that you checkpoint your jobs in 24 hour units. Submitting jobs which have short durations ensures they are more likely to start sooner. By creating a daisy chain of jobs, it is possible to overcome the seven day limit on Béluga.

1.  Modify your job submission script (or your program) so that your job can be interrupted and continued . Your program should be able to access the most recent checkpoint file. (See the example script below).
2.  Verify how many epochs (or iterations) can be carried out in a 24 hour unit.
3.  Calculate how many of these 24 hour units you will need: `n_units = n_epochs_total / n_epochs_per_24h`
4.  Use the argument `--array 1-``<n_blocs>`{=html}`%1` to ask for a chain of `n_blocs` jobs.

The job submission script will look like this:

`xargs -r -0 ls -1 -t  head -1)`

1.  Start training

if \[ -z \"\$LAST_CHECKPOINT\" \]; then

`   # $LAST_CHECKPOINT is null; start from scratch`\
`   python $SOURCEDIR/train.py --write-checkpoints-to $CHECKPOINTS ...`

else

`   python $SOURCEDIR/train.py --load-checkpoint $LAST_CHECKPOINT --write-checkpoints-to $CHECKPOINTS ...`

fi }}
