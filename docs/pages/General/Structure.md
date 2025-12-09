---
title: "Structure/en"
url: "https://docs.alliancecan.ca/wiki/Structure/en"
category: "General"
last_modified: "2022-01-21T18:35:08Z"
page_id: 8832
display_title: "Structure"
---

== Description ==

StructureStructure Homepage: http://web.stanford.edu/group/pritchardlab/structure.html is a free software package for using multilocus genotype data to investigate population structure. Its uses include inferring the presence of distinct populations, assigning individuals to populations, studying hybrid zones, identifying migrants and admixed individuals, and estimating population allele frequencies in situations where many individuals are migrants or admixed. It can be applied to most of the commonly-used genetic markers, including SNPS, microsatellites, RFLPs and AFLPs.
J.K. Pritchard, M. Stephens, and P. Donnelly. Inference of population structure using multilocus genotype data. Genetics, 155:945–959, 2000. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1461096/
M.J. Hubisz, D. Falush, M. Stephens, and J.K. Pritchard. Inferring weak population structure with the assistance of sample group information. Molecular Ecology Resources, 9(5):1322–1332, 2009. doi: https://doi.org/10.1111/j.1755-0998.2009.02591.x

== Installed versions ==

At the time of writing Structure v2.3.4 has been installed (module: structure/2.3.4).
Please look on page Available software for module structure for an up-to-date list of installed versions.

== Usage ==

When starting structure without any command-line options, it expects to find the following three files in the current working directory:
* mainparams - Example mainparams Example mainparams on the Structure Hompage
* extraparams - Example extraparamsExample extraparams  on the Structure Hompage
* and a data file. The name of the data file can be either be set the INFILE parameter in mainparams or supplied via the command line (-i).

Please refer to §7 "Running structure from the command line" in the Structure DocumentationStructure Documentation for version 2.3.4 PDF for a complete description of the available parameters and file-formats. There are also several command-line options to supply alternative filenames or override options. e.g.: -m (mainparams), -e (extraparams), -i (input file),  -o (output file).
See section §7.4 in the Structure Documentation.

Here is an example submission script:

== Running Structure in parallel: StrAuto & StructureHarvester ==
The Structure in itself is not able to run in parallel, however as population structure inference using the Evanno-method
G. Evanno, S. Regnaut, and J. Goudet. Detecting the number of clusters of individuals using the software structure: a simulation study. Molecular Ecology, 14:2611–2620, 2005. DOI: https://doi.org/10.1111/j.1365-294X.2005.02553.x
involves multiple structure-runs, the tools StrAuto
StrAuto Homepage: https://vc.popgen.org/software/strauto/
StrAuto User Guide: https://vc.popgen.org/software/strauto/strauto_doc.pdf
Chhatre, VE & Emerson KJ. StrAuto: Automation and parallelization of STRUCTURE analysis. BMC Bioinformatics (2017) 18:192. doi: http://dx.doi.org/10.1186/s12859-017-1593-0 and StructureHarvester
StructureHarvester Homepage: http://alumni.soe.ucsc.edu/~dearl/software/structureHarvester/ ; GitHub: https://github.com/dentearl/structureHarvester
Earl, Dent A. and vonHoldt, Bridgett M. STRUCTURE HARVESTER: a website and program for visualizing STRUCTURE output and implementing the Evanno method. Conservation Genetics Resources (2011) DOI: 10.1007/s12686-011-9548-7
have been developed to automate the process of setting up such runs and aggregating their results respectively.

=== Practical considerations ===
There is an example for running StrAuto jobs on HPC clusters using the Slurm Workload manager outlined in chapter 8 the StrAuto User Guide.
The example works best when the total number of Structure runs is a multiple of the number of requested tasks, as otherwise some of the allocated CPUs will sit idle
while the last Structure runs are being executed.  This may lead to a significant waste of computing resources.

Moreover, the requested maximum job time needs to be sufficiently large to accommodate multiple subsequent Structure runs.
While this can be a sensible choice for relatively short runs, jobs with long walltimes will typically have to wait longer until they are dispatched due to
job scheduling policies.
Therefore in cases where an individual Structure run takes more than just a few hours to finish, we recommend submitting each Structure run as an individual job to Slurm.

=== Running a set of longer Structure runs ===

The script create_strauto_slurm_scripts.py shown at the bottom of this section
is designed to help you run Structure jobs on Compute Canada HPC clusters that use the Slurm Workload manager.
It requires Structure, StrAuto, and StructureHarvester.

Usage:

* Place the create_strauto_slurm_scripts.py script shown below into a directory along with:
:* strauto_1.py and input.py from StrAuto.
:* structureHarvester.py and harvesterCore.py from StructureHarvester.
:* The file with the Structure dataset e.g. my_dataset.str.
* Make create_strauto_slurm_scripts.py executable with:

:* You should now have the following files:

* Edit the settings in input.py as described in the StrAuto User Manual.
:* Make sure to set the option parallel = True (question 23).
* Adjust the parameters in this file (lines 65-70):
:* Set max_jobtime to a duration where you can be reasonably sure that no individual Structure run takes longer than this.
:* In cases where a user can submit under multiple Slurm-accounts slurm_account can be used to specify under which account to submit.
:* To avoid overloading the Scheduler, the submission helper script delays each submission by a time defined by submit_delay.
* Run the following commands:

* Submit the jobs with:

* After all jobs have completed, run bash post_strauto.sh to aggregate the results and run StructureHarvester.

==== The 'create_strauto_slurm_scripts.py' script ====

== References ==