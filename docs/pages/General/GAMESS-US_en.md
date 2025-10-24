---
title: "GAMESS-US/en"
url: "https://docs.alliancecan.ca/wiki/GAMESS-US/en"
category: "General"
last_modified: "2024-07-12T14:55:04Z"
page_id: 6511
display_title: "GAMESS-US"
---

`<languages />`{=html}

The General Atomic and Molecular Electronic Structure System (GAMESS) [^1] is a general *ab initio* quantum chemistry package.

## Running GAMESS {#running_gamess}

### Job submission {#job_submission}

Compute Canada clusters use the Slurm scheduler. For more about submitting and monitoring jobs, see [Running jobs](https://docs.alliancecan.ca/Running_jobs "Running jobs"){.wikilink}.

The first step is to prepare a GAMESS input file containing the molecular geometry and a specification of the calculation to be carried out. Please refer to the GAMESS Documentation [^2] and particularly Chapter 2 \"Input Description\"[^3] for a description the file format and keywords.

Besides your input file (in our example, \"name.inp\"), you have to prepare a job script to define the compute resources for the job. Input file and job script must be in the same directory.

```{=mediawiki}
{{File
  |name=gamess_job.sh
  |lang="sh"
  |contents=
#!/bin/bash
#SBATCH --cpus-per-task=1       # Number of CPUs
#SBATCH --mem-per-cpu=4000M     # memory per CPU in MB
#SBATCH --time=0-00:30          # time (DD-HH:MM)

## Directory for GAMESS supplementary output files ($USERSCR):
#export USERSCR=$SCRATCH

## Directory for GAMESS temporary binary files ($SCR):
## Uncomment the following two lines to use /scratch instead of local disk
#export SCR="$SCRATCH/gamess_${SLURM_JOB_ID}/"
#mkdir -p $SCR

module load gamess-us/20170420-R1
export SLURM_CPUS_PER_TASK      # rungms will use this
rungms name.inp  &>  name.out
}}
```
Use the following command to submit the job to the scheduler:

`  sbatch gamess_job.sh`

### Scratch files {#scratch_files}

By default, temporary binary files (scratch files) will be written to local disk on the compute node (`$SLURM_TMPDIR`) as we expect this to give the best performance. Please be aware that the data in `$SLURM_TMPDIR` will be **deleted** after the job finishes. If there is insufficient space on local disk, you can use /scratch instead by setting the `SCR` environment variable as shown in the comments in the example above.

Supplementary output files are written to a location defined by the `USERSCR` environment variable. By default this is the user\'s `$SCRATCH` directory.

  Description                         Environment Variable   Default location
  ----------------------------------- ---------------------- ----------------------------------------
  GAMESS temporary binary files       `SCR`                  `$SLURM_TMPDIR` (node-local storage)
  GAMESS supplementary output files   `USERSCR`              `$SCRATCH` (user\'s SCRATCH directory)

### Running GAMESS on multiple CPUs {#running_gamess_on_multiple_cpus}

GAMESS calculations can make use of more than one CPU. The number of CPUs available for a calculation is determined by the `--cpus-per-task` setting in the job script.

As GAMESS has been built using [sockets](https://en.wikipedia.org/wiki/Unix_domain_socket) for parallelization, it can only use CPU cores that are located on the same compute node. Therefore the maximum number of CPU cores that can be used for a job is dictated by the size of the nodes in the cluster, e.g. 32 CPU cores per node on [Graham](https://docs.alliancecan.ca/Graham "Graham"){.wikilink}.

Quantum chemistry calculations are known to not scale well to large numbers of CPUs as compared to e.g. classical molecular mechanics, which means that they can\'t use large numbers of CPUs efficiently. Exactly how many CPUs can be used efficiently depends on the number of atoms, the number of basis functions, and the level of theory.

To determine a reasonable number of CPUs to use, one needs to run a scaling test\-\-- That is, run the same input file using different numbers of CPUs and compare the execution times. Ideally the execution time should be half as long when using twice as many CPUs. Obviously it is not a good use of resources if a calculation runs (for example) only 30% faster when the number of CPUs is doubled. It is even possible for certain calculations to run slower when increasing the number of CPUs.

### Memory

Quantum chemistry calculations are often \"memory bound\"\-\-- meaning that larger molecules at high level of theory need a lot of memory (RAM), often more than is available in a typical computer. Therefore packages like GAMESS use disk storage (SCRATCH) to store intermediate results to free up memory, reading them back from disk later in the calculation.

Even our fastest SCRATCH storage is several orders of magnitudes slower than the memory, so you should make sure to assign sufficient memory to GAMESS. This is a two-step process:

1\. Request memory for the job in the submission script. Using `--mem-per-cpu=4000M` is a reasonable value, since it matches the memory-to-CPU ratio on the base nodes. Requesting more than that may cause the job to wait to be scheduled on a large-memory node.

2\. In the `$SYSTEM` group of the input file, define the `MWORDS` and `MEMDDI` options. This tells GAMESS how much memory it is allowed to use.

- `MWORDS` is the maximum replicated memory which a job can use, on every core. This is given in units of 1,000,000 words (as opposed to 1024\*1024 words), and a word is defined as 64 bits = 8 bytes.
- `MEMDDI` is the grand total memory needed for the distributed data interface (DDI) storage, given in units of 1,000,000 words. The memory required on each processor core for a run using `p` CPU-cores is therefore `MEMDDI/p + MWORDS`.

Please refer to the `$SYSTEM group` section in the GAMESS documentation[^4] if you want more details.

It is important to leave a few hundred MB of memory between the memory requested from the scheduler and the memory that GAMESS is allowed to use, as a safety margin. If a job\'s output is incomplete and the `slurm-{JOBID}.out` file contains a message like \"slurmstepd: error: Exceeded step/job memory limit at some point\",then Slurm has terminated the job for trying to use more memory than was requested. In that case one needs to either reduce the `MWORDS` or `MEMDDI` in the input file or increase the `--mem-per-cpu` in the submission script.

## References

<references />

[^1]: GAMESS Homepage: <http://www.msg.ameslab.gov/gamess/>

[^2]: Documentation: <http://www.msg.ameslab.gov/gamess/documentation.html>

[^3]: Input Description (PDF): <http://www.msg.ameslab.gov/gamess/GAMESS_Manual/input.pdf>

[^4]:
