---
title: "BUSCO/en"
url: "https://docs.alliancecan.ca/wiki/BUSCO/en"
category: "General"
last_modified: "2024-05-01T23:07:01Z"
page_id: 9850
display_title: "BUSCO"
---

BUSCO (Benchmarking sets of Universal Single-Copy Orthologs) is an application for assessing genome assembly and annotation completeness.

For more information, see the user manual.

== Available versions ==
Recent versions are available as wheels. Older versions are available as a module; please see the Modules section below.

To see the latest available version, run

== Python wheel ==
=== Installation ===
1. Load the necessary modules.

2. Create the virtual environment.

3. Install the wheel and its dependencies.
1.81 pandas2.1.0 busco5.5.0
}}

4. Validate the installation.

5.  Freeze the environment and requirements set. To use the requirements text file, see the bash submission script shown at point 8.

=== Usage ===
==== Datasets ====
6. You must pre-download any datasets from BUSCO data before submitting your job.

You can access the available datasets in your terminal by typing busco --list-datasets.

You have two options to download datasets:
*use the busco command,
*use the wget command.

===== 6.1  Using the busco command =====
This is the preferred option. Type this command in your working directory to download a particular dataset, for example

It is also possible to do a bulk download by replacing the dataset name by the following arguments: all, prokaryota, eukaryota, or virus, for example

This will
::1. create a BUSCO directory hierarchy for the datasets,
::2. download the appropriate datasets,
::3. decompress the file(s),
::4. if you download multiple files, they will all be automatically added to the lineages directory.

The hierarchy will look like this:

* busco_downloads/

::* information/

::::lineages_list.2021-12-14.txt

::* lineages/

::::bacteria_odb10

::::actinobacteria_class_odb10

::::actinobacteria_phylum_odb10

::* placement_files/

::::list_of_reference_markers.archaea_odb10.2019-12-16.txt

Doing so, all your lineage files should be in busco_downloads/lineages/. When referring to --download_path busco_downloads/ in the BUSCO command line, it will know where to find the lineage dataset argument --lineage_dataset bacteria_odb10. If the busco_download  directory is not in your working directory, you will need to provide the full path.

=====6.2 Using the wget command =====

All files must be decompressed with tar -xvf file.tar.gz.

==== Test ====
7. Download a genome file.

8. Run.

Command to run a single genome:

Command to run multiple genomes that would be saved in the genome directory (in this example, the genome/ folder would need to be in the current directory; otherwise, you need to provide the full path):

The single genome command should take less than 60 seconds to complete. Production runs which take longer must be submitted to the scheduler.

===== BUSCO tips =====

Specify --in genome.fna for single file analysis.

Specify --in genome/ for multiple files analysis.

===== Slurm tips =====
Specify --offline to avoid using the internet.

Specify --cpu to $SLURM_CPUS_PER_TASK in your job submission script to use the number of CPUs allocated.

Specify --restart to restart from a partial run.

====Job submission====

Here you have an example of a submission script. You can submit as so: sbatch run_busco.sh.

====Augustus parameters====
9. Advanced users may want to use Augustus parameters: --augustus_parameters="--yourAugustusParameter".

*Copy the Augustus config directory to a writable location.

*Make sure to define the AUGUSTUS_CONFIG_PATH environment variable.
$HOME/augustus_config}}

====SEPP parameters====
10. To use SEPP parameters, you need to install SEPP locally in your virtual environment. This should be done from the login node.

10.1. Activate your BUSCO virtual environment.

10.2. Install DendroPy.

10.3. Install SEPP.

10.4. Validate the installation.

10.5. Because SEPP is installed locally, you cannot create the virtual environment as described in the previous submission script. To activate your local virtual environment, simply add the following command immediately under the line to load the module:

== Modules ==

1. Load the necessary modules.

This will also load modules for Augustus, BLAST+, HMMER and some other
software packages that BUSCO relies upon.

2. Copy the configuration file.

or

3. Edit the configuration file. The locations of external tools are all specified in the last section, which is shown below:

4. Copy the Augustus config directory to a writable location.

5. Check that it runs.

$HOME/busco_config.ini
|export AUGUSTUS_CONFIG_PATH$HOME/augustus_config
|run_BUSCO.py --in $EBROOTBUSCO/sample_data/target.fa --out TEST --lineage_path $EBROOTBUSCO/sample_data/example --mode genome
}}

The run_BUSCO.py command should take less than 60 seconds to complete.
Production runs which take longer should be submitted to the scheduler.

= Troubleshooting =
== Cannot write to Augustus config path ==
Make sure you have copied the config directory to a writable location and exported the AUGUSTUS_CONFIG_PATH variable.