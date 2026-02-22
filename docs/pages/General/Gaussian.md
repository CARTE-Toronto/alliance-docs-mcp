---
title: "Gaussian/en"
url: "https://docs.alliancecan.ca/wiki/Gaussian/en"
category: "General"
last_modified: "2026-02-13T17:19:49Z"
page_id: 3737
display_title: "Gaussian"
---

See also Gaussian error messages.
Gaussian is a computational chemistry application produced by Gaussian, Inc.

== Limitations ==

We currently support Gaussian on Nibi and Fir.

Cluster/network parallel execution of Gaussian, also known as "Linda parallelism", is not supported at any of our national systems.
Only "shared-memory multiprocessor parallel execution" is supported.
Therefore no Gaussian job can use more than a single compute node.

== License agreement ==

In order to use Gaussian you must agree to certain conditions. Please  contact support with a copy of the following statement:
# I am not a member of a research group developing software competitive to Gaussian.
# I will not copy the Gaussian software, nor make it available to anyone else.
# I will properly acknowledge Gaussian Inc. and the Alliance in publications.
# I will notify the Alliance of any change in the above acknowledgement.
If you are a sponsored user, your sponsor (PI) must also have such a statement on file with us.

We will then grant you access to Gaussian.

==Running Gaussian on Fir and Nibi==
The gaussian module is installed on Nibi and Fir. To check what versions are available use the module spider command as follows:

 [name@server $] module spider gaussian

For module commands, please see Using modules.

===Job submission===
The national clusters use the Slurm scheduler; for details about submitting jobs, see Running jobs.

Since only the "shared-memory multiprocessor" parallel version of Gaussian is supported, your jobs can use only one node and up to the maximum cores per node. However due to the scalability of Gaussian, we recommend that you use no more than 32 CPUs per job unless you have good evidence that you can use them efficiently!  The new clusters Nibi and Fir have 192 CPUs per node.  Please do not simply run full-node Gaussian jobs on these clusters; it will be inefficient.  If your jobs are limited by the amount of available memory on a single node, be aware that there are a few nodes at each site with more than the usual amount of memory.  Please refer to the pages  Fir and Nibi for the number and capacity of such nodes.

Besides your input file (in our example, "name.com"), you have to prepare a job script to define the compute resources for the job; both input file and job script must be in the same directory.

There are two options to run your Gaussian job on the clusters, based on the location of the default runtime files and the job size.

====G16 (G09, G03)====

This option will save the default runtime files (unnamed .rwf, .inp, .d2e, .int, .skr files) to /scratch/username/jobid/. Those files will stay there when the job is unfinished or failed for whatever reason, you could locate the .rwf file for restart purpose later.

The following example is a G16 job script:

Note that for coherence, we use the same name for each files, changing only the extension (name.sh, name.com, name.log).

To use Gaussian 09 or Gaussian 03, simply modify the module load gaussian/g16.c01 to gaussian/g09.e01 or gaussian/g03.d01, and change G16 to G09 or G03. You can modify the --mem, --time, --cpus-per-task to match your job's requirements for compute resources.

====g16 (g09, g03)====

This option will save the default runtime files (unnamed .rwf, .inp, .d2e, .int, .skr files) temporarily in $SLURM_TMPDIR (/localscratch/username.jobid.0/) on the compute node where the job was scheduled to run. Gaussian jobs will run faster when using the /localscratch. These files will be automatically removed by the scheduler when the job finishes, whether successfully or not. If you plan to use the .rwf file to restart the job in a later time, you must explicitly specify and name your own .rwf file in the Gaussian input file.

/localscratch is ~3TB shared by all jobs running on the same node. If your job files would be bigger than or close to that size range, you would instead use the G16 (G09, G03) option.

The following example is a g16 job script:

====Submit the job====
 sbatch mysub.sh

=== Interactive jobs ===
You can run interactive Gaussian job for testing purpose on the clusters. It's not a good practice to run interactive Gaussian jobs on a login node. You can start an interactive session on a compute node with salloc, the example for an hour, 8 cpus and 10G memory Gaussian job is like
Goto the input file directory first, then use salloc command:
1:0:0 --cpus-per-task8 --mem10g}}

Then use either

or

=== Restart jobs ===
Gaussian jobs can always be restarted from the previous rwf file.

Geometry optimization can be restarted from the chk file as usual.
One-step computation, such as Analytic frequency calculations, including properties like ROA and VCD with ONIOM; CCSD and EOM-CCSD calculations; NMR; Polar=OptRot; CID, CISD, CCD, QCISD and BD energies, can be restarted from the rwf file.

To restart a job from previous rwf file, you need to know the location of this rwf file from your previous run.

The restart input is simple: first you need to specify %rwf path to the previous rwf file, secondly change the keywords line to be #p restart, then leave a blank line at the end.

A sample restart input is like:

===Examples===
An example input file and the run scripts *.sh can be found in
/opt/software/gaussian/version/examples/
where version is either g03.d10, g09.e01, or g16.b01

== Notes ==
# NBO7 is included in g16.c01 version only, both nbo6 and nbo7 keywords will run NBO7 in g16.c01
# NBO6 is available in g09.e01 and g16.b01 versions.
# You can find a webinar slides:  Running Gaussian16 and NBO7 effectively on Nibi and Fir (2026)

== Errors ==
Some of the error messages produced by Gaussian have been collected, with suggestions for their resolution. See Gaussian error messages.