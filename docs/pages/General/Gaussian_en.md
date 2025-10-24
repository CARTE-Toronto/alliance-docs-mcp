---
title: "Gaussian/en"
url: "https://docs.alliancecan.ca/wiki/Gaussian/en"
category: "General"
last_modified: "2025-09-05T19:26:33Z"
page_id: 3737
display_title: "Gaussian"
---

`<languages />`{=html}

*See also [Gaussian error messages](https://docs.alliancecan.ca/Gaussian_error_messages "Gaussian error messages"){.wikilink}.*\
\
Gaussian is a computational chemistry application produced by [Gaussian, Inc.](http://gaussian.com/)

## Limitations

We currently support Gaussian on [Nibi](https://docs.alliancecan.ca/Nibi "Nibi"){.wikilink} and [Fir](https://docs.alliancecan.ca/Fir "Fir"){.wikilink}.

[Cluster/network parallel execution](https://gaussian.com/running/?tabid=4) of Gaussian, also known as \"Linda parallelism\", is not supported at any of our national systems. Only [\"shared-memory multiprocessor parallel execution\"](https://gaussian.com/running/?tabid=4) is supported.\
Therefore no Gaussian job can use more than a single compute node.

## License agreement {#license_agreement}

In order to use Gaussian you must agree to certain conditions. Please [ contact support](https://docs.alliancecan.ca/Technical_support " contact support"){.wikilink} with a copy of the following statement:

1.  I am not a member of a research group developing software competitive to Gaussian.
2.  I will not copy the Gaussian software, nor make it available to anyone else.
3.  I will properly acknowledge Gaussian Inc. and [the Alliance](https://alliancecan.ca/en/services/advanced-research-computing/acknowledging-alliance) in publications.
4.  I will notify the Alliance of any change in the above acknowledgement.

If you are a sponsored user, your sponsor (PI) must also have such a statement on file with us.

We will then grant you access to Gaussian.

## Running Gaussian on Fir and Nibi {#running_gaussian_on_fir_and_nibi}

The `gaussian` module is installed on [Nibi](https://docs.alliancecan.ca/Nibi "Nibi"){.wikilink} and [Fir](https://docs.alliancecan.ca/Fir "Fir"){.wikilink}. To check what versions are available use the `module spider` command as follows:

`[name@server $] module spider gaussian`

For module commands, please see [Using modules](https://docs.alliancecan.ca/Utiliser_des_modules/en "Using modules"){.wikilink}.

### Job submission {#job_submission}

The national clusters use the Slurm scheduler; for details about submitting jobs, see [Running jobs](https://docs.alliancecan.ca/Running_jobs "Running jobs"){.wikilink}.

Since only the \"shared-memory multiprocessor\" parallel version of Gaussian is supported, your jobs can use only one node and up to the maximum cores per node. However due to the scalability of Gaussian, we recommend that you `<em>`{=html}use no more than 32 CPUs per job unless you have good evidence that you can use them efficiently!`</em>`{=html} The new clusters Nibi and Fir have 192 CPUs per node. Please do not simply run full-node Gaussian jobs on these clusters; it will be inefficient. If your jobs are limited by the amount of available memory on a single node, be aware that there are a few nodes at each site with more than the usual amount of memory. Please refer to the pages [Fir](https://docs.alliancecan.ca/Fir/en#Node_characteristics "Fir"){.wikilink} and [Nibi](https://docs.alliancecan.ca/Nibi/en#Node_characteristics "Nibi"){.wikilink} for the number and capacity of such nodes.

Besides your input file (in our example, \"name.com\"), you have to prepare a job script to define the compute resources for the job; both input file and job script must be in the same directory.

There are two options to run your Gaussian job on the clusters, based on the location of the default runtime files and the job size.

#### G16 (G09, G03) {#g16_g09_g03}

This option will save the default runtime files (unnamed .rwf, .inp, .d2e, .int, .skr files) to /scratch/username/jobid/. Those files will stay there when the job is unfinished or failed for whatever reason, you could locate the .rwf file for restart purpose later.

The following example is a G16 job script:

Note that for coherence, we use the same name for each files, changing only the extension (name.sh, name.com, name.log).

To use Gaussian 09 or Gaussian 03, simply modify the module load gaussian/g16.c01 to gaussian/g09.e01 or gaussian/g03.d01, and change G16 to G09 or G03. You can modify the \--mem, \--time, \--cpus-per-task to match your job\'s requirements for compute resources.

#### g16 (g09, g03) {#g16_g09_g03_1}

This option will save the default runtime files (unnamed .rwf, .inp, .d2e, .int, .skr files) temporarily in \$SLURM_TMPDIR (/localscratch/username.jobid.0/) on the compute node where the job was scheduled to. The files will be removed by the scheduler when a job is done (successful or not). If you do not expect to use the .rwf file to restart in a later time, you can use this option.

/localscratch is \~800G shared by all jobs running on the same node. If your job files would be bigger than or close to that size range, you would instead use the G16 (G09, G03) option.

The following example is a g16 job script:

#### Submit the job {#submit_the_job}

`sbatch mysub.sh`

### Interactive jobs {#interactive_jobs}

You can run interactive Gaussian job for testing purpose on the clusters. It\'s not a good practice to run interactive Gaussian jobs on a login node. You can start an interactive session on a compute node with salloc, the example for an hour, 8 cpus and 10G memory Gaussian job is like Goto the input file directory first, then use salloc command: 1:0:0 \--cpus-per-task8 \--mem10g}}

Then use either

or

### Restart jobs {#restart_jobs}

Gaussian jobs can always be restarted from the previous `rwf` file.

Geometry optimization can be restarted from the `chk` file as usual. One-step computation, such as Analytic frequency calculations, including properties like ROA and VCD with ONIOM; CCSD and EOM-CCSD calculations; NMR; Polar=OptRot; CID, CISD, CCD, QCISD and BD energies, can be restarted from the `rwf` file.

To restart a job from previous `rwf` file, you need to know the location of this `rwf` file from your previous run.

The restart input is simple: first you need to specify %rwf path to the previous `rwf` file, secondly change the keywords line to be #p restart, then leave a blank line at the end.

A sample restart input is like:

### Examples

An example input file and the run scripts `*.sh` can be found in `/opt/software/gaussian/version/examples/` where version is either g03.d10, g09.e01, or g16.b01

## Notes

1.  NBO7 is included in g16.c01 version only, both nbo6 and nbo7 keywords will run NBO7 in g16.c01
2.  NBO6 is available in g09.e01 and g16.b01 versions.
3.  You can watch a recorded webinar/tutorial: [Gaussian16 and NBO7 on Graham and Cedar](https://www.youtube.com/watch?v=xpBhPnRbeQo) (2022)

## Errors

Some of the error messages produced by Gaussian have been collected, with suggestions for their resolution. See [Gaussian error messages](https://docs.alliancecan.ca/Gaussian_error_messages "Gaussian error messages"){.wikilink}.
