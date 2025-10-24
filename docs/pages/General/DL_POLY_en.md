---
title: "DL POLY/en"
url: "https://docs.alliancecan.ca/wiki/DL_POLY/en"
category: "General"
last_modified: "2025-02-20T15:28:43Z"
page_id: 11152
display_title: "DL POLY"
---

`<languages />`{=html}

# General

DL_POLY is a general purpose classical molecular dynamics (MD) simulation software. It provides scalable performance from a single processor workstation to a high performance parallel computer. DL_POLY_4 offers fully parallel I/O as well as a NetCDF alternative to the default ASCII trajectory file.

There is a mailing list [here.](https://www.jiscmail.ac.uk/cgi-bin/webadmin?A0=DLPOLY)

# License limitations {#license_limitations}

**DL_POLY** is now [open source](https://gitlab.com/DL%20POLY%20Classic/dl%20poly) and it does not require registration. A new module **dl_poly4/5.1.0** is already installed under **StdEnv/2023** and it is accessible for all users. However, if you would like to use the previous versions (**dl_poly4/4.10.0** and/or **dl_poly4/4.08**), you should contact [ support](https://docs.alliancecan.ca/Technical_support " support"){.wikilink} and ask to be added to the POSIX group that controls access to DL_POLY4. There is no need to register on DL_POLY website.

# Modules

To see which versions of DL_POLY are installed on our systems, run `module spider dl_poly4`. See [Using modules](https://docs.alliancecan.ca/Using_modules "Using modules"){.wikilink} for more about `module` subcommands.

To load the version **5.x**, use:

`module load StdEnv/2023 intel/2023.2.1 openmpi/4.1.5 dl_poly4/5.1.0`

To load the previous version 4.10.0, use:

`module load StdEnv/2023 intel/2020.1.217 openmpi/4.0.3 dl_poly4/4.10.0`

Note that this version requires to be added to a POSIX group as explained above in [ License limitations](https://docs.alliancecan.ca/#License_limitations " License limitations"){.wikilink}.

We do not currently provide a module for the Java GUI interface.

# Scripts and examples {#scripts_and_examples}

The input files shown below (CONTROL and FIELD) were taken from example TEST01 that can be downloaded from the page of [DL_POLY examples](https://docs.alliancecan.ca/ftp://ftp.dl.ac.uk/ccp5/DL_POLY/DL_POLY_4.0/DATA/).

To start a simulation, one must have at least three files:

- **CONFIG**: simulation box (atomic coordinates)
- **FIELD**: force field parameters
- **CONTROL**: simulation parameters (time step, number of MD steps, simulation ensemble, \...etc.)

`<tabs>`{=html} `<tab name="CONTROL">`{=html}

`</tab>`{=html} `<tab name="FIELD">`{=html}

`</tab>`{=html} `<tab name="Serial job">`{=html} ``{{File
  |name=run_serial_dlp.sh
  |lang="bash"
  |contents=
#!/bin/bash

#SBATCH --account=def-someuser
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2500M      # memory; default unit is megabytes.
#SBATCH --time=0-00:30           # time (DD-HH:MM).

# Load the module:

module load StdEnv/2023  
module load intel/2023.2.1  openmpi/4.1.5 dl_poly4/5.1.0

echo "Starting run at: `date`"

dlp_exec=DLPOLY.Z

${dlp_exec}

echo "Program finished with exit code $? at: `date`"
}}``{=mediawiki} `</tab>`{=html} `<tab name="MPI job">`{=html} ``{{File
  |name=run_mpi_dlp.sh
  |lang="bash"
  |contents=
#!/bin/bash

#SBATCH --account=def-someuser
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=2500M      # memory; default unit is megabytes.
#SBATCH --time=0-00:30           # time (DD-HH:MM).

# Load the module:

module load StdEnv/2023  
module load intel/2023.2.1  openmpi/4.1.5 dl_poly4/5.1.0

echo "Starting run at: `date`"

dlp_exec=DLPOLY.Z

srun ${dlp_exec}

echo "Program finished with exit code $? at: `date`"
}}``{=mediawiki} `</tab>`{=html} `</tabs>`{=html}

# Related software {#related_software}

- [VMD](https://docs.alliancecan.ca/VMD "VMD"){.wikilink}
- [LAMMPS](https://docs.alliancecan.ca/LAMMPS "LAMMPS"){.wikilink}
