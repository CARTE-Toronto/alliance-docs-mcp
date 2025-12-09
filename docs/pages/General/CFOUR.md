---
title: "CFOUR/en"
url: "https://docs.alliancecan.ca/wiki/CFOUR/en"
category: "General"
last_modified: "2024-04-03T17:28:45Z"
page_id: 9101
display_title: "CFOUR"
---

`<languages/>`{=html}

# Introduction

\"`<b>`{=html}CFOUR`</b>`{=html} (Coupled-Cluster techniques for Computational Chemistry) is a program package for performing high-level quantum chemical calculations on atoms and molecules. The major strength of the program suite is its rather sophisticated arsenal of high-level `<i>`{=html}ab-initio`</i>`{=html} methods for the calculation of atomic and molecular properties. Virtually all approaches based on Møller-Plesset (MP) perturbation theory and the coupled-cluster approximation (CC) are available; most of these have complementary analytic derivative approaches within the package as well.\"

\"`<b>`{=html}CFOUR`</b>`{=html} is not a commercial code. It is rather a program that is undergoing development; new techniques and improvements are constantly being made.\" See [the CFOUR web site](http://slater.chemie.uni-mainz.de/cfour/index.php?n=Main.HomePage) for more information.

# License limitations {#license_limitations}

The Alliance has signed a [license](http://slater.chemie.uni-mainz.de/cfour/index.php?n=Main.Download) agreement with [Prof. Dr. J. Gauss](https://www.tc.uni-mainz.de/prof-dr-juergen-gauss/) who acts for the developers of the CFOUR Software.

In order to use the current installed version on the Alliance systems, each user must agree to certain conditions. Please [ contact support](https://docs.alliancecan.ca/Technical_support " contact support"){.wikilink} with a copy of the following statement:

1.  I will use CFOUR only for academic research.
2.  I will not copy the CFOUR software, nor make it available to anyone else.
3.  I will properly acknowledge original papers related to CFOUR and to the Alliance in my publications (see the license form for more details).
4.  I understand that the agreement for using CFOUR can be terminated by one of the parties: CFOUR developers or the Alliance.
5.  I will notify the Alliance of any change in the above acknowledgement.

When your statement is received, we will allow you to access the program.

# Module

You can access the MPI version of CFOUR by loading a [module](https://docs.alliancecan.ca/Utiliser_des_modules/en "module"){.wikilink}.

``` bash
module load intel/2023.2.1  openmpi/4.1.5 cfour-mpi/2.1
```

For the serial version, use:

``` bash
module load intel/2023.2.1 cfour/2.1
```

There is a mailing list as a forum for user experiences with the CFOUR program system. For how to subscribe and other information, see [this page](http://slater.chemie.uni-mainz.de/cfour/index.php?n=Main.MailingList).

## Examples and job scripts {#examples_and_job_scripts}

To run CFOUR, you need to have at least the input file [ZMAT](http://slater.chemie.uni-mainz.de/cfour/index.php?n=Main.InputFileZMAT) with all information concerning geometry, requested quantum-chemical method, basis set, etc. The second file is [GENBAS](http://slater.chemie.uni-mainz.de/cfour/index.php?n=Main.Basis-setFileGENBAS) that contains the required information for the basis sets available to the user. If GENBAS is not present in the directory from where you start your job, CFOUR will create a symlink and use the existing file provided by the module. The file is located at: `$EBROOTCFOUR/basis/GENBAS`.

`<tabs>`{=html} `<tab name="INPUT">`{=html}

`</tab>`{=html}

`<tab name="Serial job">`{=html} ``{{File
  |name=run_cfour_serial.sh
  |lang="bash"
  |contents=
#!/bin/bash
#SBATCH --account=def-someacct   # replace this with your own account
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2500M      # memory; default unit is megabytes.
#SBATCH --time=0-00:30           # time (DD-HH:MM).

# Load the module:

module load intel/2023.2.1 cfour/2.1

echo "Starting run at: `date`"

CFOUROUTPUT="cfour-output.txt"
export CFOUR_NUM_CORES=1

xcfour > ${CFOUROUTPUT} 

# Clean the symlink:
if [[ -L "GENBAS" ]]; then unlink GENBAS; fi

echo "Program finished with exit code $? at: `date`"
}}``{=mediawiki} `</tab>`{=html}

`<tab name="MPI job">`{=html} ``{{File
  |name=run-cfour-mpi.sh
  |lang="bash"
  |contents=
#!/bin/bash
#SBATCH --account=def-someacct   # replace this with your own account
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=2500M      # memory; default unit is megabytes.
#SBATCH --time=0-00:30           # time (DD-HH:MM).

# Load the module:

module load intel/2023.2.1  openmpi/4.1.5 cfour-mpi/2.1

echo "Starting run at: `date`"

CFOUROUTPUT="cfour-output.txt"
export CFOUR_NUM_CORES=${SLURM_NTASKS}

xcfour > ${CFOUROUTPUT} 

# Clean the symlink:
if [[ -L "GENBAS" ]]; then unlink GENBAS; fi

echo "Program finished with exit code $? at: `date`"
}}``{=mediawiki} `</tab>`{=html} `</tabs>`{=html}

# Related links {#related_links}

- [Manual](http://slater.chemie.uni-mainz.de/cfour/index.php?n=Main.Manual)
- [Features](http://slater.chemie.uni-mainz.de/cfour/index.php?n=Main.Features)
