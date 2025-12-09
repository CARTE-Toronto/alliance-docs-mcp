---
title: "MIKE/en"
url: "https://docs.alliancecan.ca/wiki/MIKE/en"
category: "General"
last_modified: "2025-09-03T12:24:42Z"
page_id: 21986
display_title: "MIKE"
---

`<languages />`{=html}

[MIKE powered by DHI](https://www.mikepoweredbydhi.com/) is a a hydraulic and hydrological modeling software package.

## License requirements {#license_requirements}

MIKE is a commercial product and each user needs to supply their own license.

In order for you to use it on our HPC clusters, you will need to contact MIKE Customer Care at: <mike@dhigroup.com> and confirm that you have

- an *internet license*, and
- a download link for the *Linux version* of MIKE.

## Installation

You need to download the installation archives for Linux.

The following instructions assume that the installation archives are in one Zip-file (MIKE 2025 and newer) or three `*.tgz` files (MIKE 2024 and older): `<tabs>`{=html} `<tab name="MIKE 2025">`{=html}

- `MIKE_Zero_2025_rhel9.zip`

`</tab>`{=html} `<tab name="MIKE 2024">`{=html}

- `MIKE_Zero_2024_rhel9_Update_1.tgz`
- `MIKE_Zero_2024_Tools_rhel9_Update_1.tgz`
- `MIKE_Zero_2024_Examples_Update_1.tgz`

`</tab>`{=html} `<tab name="MIKE 2023">`{=html}

- `MIKE_Zero_2023_rhel7_22.11.05.tgz`
- `MIKE_Zero_2023_Tools_rhel7_22.11.05.tgz`
- `MIKE_Zero_2023_Examples.tgz`

`</tab>`{=html} `<tab name="MIKE 2022">`{=html}

- `MIKE_Zero_2022_rhel7_Update_1.tgz`
- `MIKE_Zero_2022_Tools_rhel7_Update_1.tgz`
- `MIKE_Zero_2022_Examples_Update_1.tgz`

`</tab>`{=html} `</tabs>`{=html}

1\. Create a directory `~/scratch/MIKE_TGZ` and upload the installation archive(s) to that location.

2\. MIKE was compiled with the Intel MPI library, therefore you must load a matching `intelmpi` module.

`<tabs>`{=html} `<tab name="MIKE 2024 and 2025">`{=html}

`module load StdEnv/2023 intel/2023.2.1 intelmpi/2021.9.0`

`</tab>`{=html} `<tab name="MIKE 2023">`{=html}

`module load StdEnv/2020  intel/2021.2.0  intelmpi/2021.2.0`

`</tab>`{=html} `<tab name="MIKE 2022">`{=html}

`module load StdEnv/2020  intel/2020.1.217  intelmpi/2019.7.217`

`</tab>`{=html} `</tabs>`{=html}

3\. Run the following commands depending on the version of MIKE. They will extract the archives, run the \`install.sh\` installation scripts for each component and then [Patch the binaries](https://docs.alliancecan.ca/Installing_software_in_your_home_directory#Installing_binary_packages "Patch the binaries"){.wikilink} so that they can find the dynamic libraries of Intel MPI.

`<tabs>`{=html} `<tab name="MIKE 2025">`{=html}

`export MIKE_TGZ="$HOME/scratch/MIKE_TGZ"`\
`export MIKE_HOME="$HOME/MIKE/2025"`\
\
`cd $MIKE_TGZ`\
`unzip -j  MIKE_Zero_2025_rhel9.zip`\
`tar -xzf MIKE_Common_2025_rhel9.tgz`\
`tar -xzf MIKE_Zero_2025_rhel9.tgz`\
`tar -xzf MIKE_Zero_2025_Tools_rhel9.tgz`\
`tar -xzf MIKE_Zero_2025_Examples.tgz`\
\
`cd $MIKE_TGZ/MIKE_Common_2025_rhel9`\
`sed -i 's/ cp -rp / cp -r /' install.sh`\
`sh install.sh --eula --install-path "$MIKE_HOME" --license-server 127.0.0.1`\
`cd $MIKE_TGZ/MIKE_Zero_2025_rhel9`\
`sed -i 's/ cp -rp / cp -r /' install.sh`\
`sh install.sh --eula --install-path "$MIKE_HOME"`\
`cd $MIKE_TGZ/MIKE_Zero_2025_Tools_rhel9`\
`sed -i 's/ cp -rp / cp -r /' install.sh`\
`sh install.sh --eula --install-path "$MIKE_HOME"`\
`cd $MIKE_TGZ/MIKE_Zero_2025_Examples`\
`sed -i 's/ cp -rp / cp -r /' install.sh`\
`sh install.sh --eula --install-path "$MIKE_HOME"`\
\
`module load StdEnv/2023 intel/2023.2.1 intelmpi/2021.9.0`\
`setrpaths.sh --path "$MIKE_HOME/bin"  --add_origin  \`\
`    --add_path="$EBROOTIMPI/mpi/latest/lib/release:$EBROOTIMPI/mpi/latest/lib"`

`</tab>`{=html} `<tab name="MIKE 2024">`{=html}

`export MIKE_TGZ="$HOME/scratch/MIKE_TGZ"`\
`export MIKE_HOME="$HOME/MIKE/2024"`\
\
`cd $MIKE_TGZ`\
`tar -xzf MIKE_Zero_2024_rhel9_Update_1.tgz`\
`tar -xzf MIKE_Zero_2024_Tools_rhel9_Update_1.tgz`\
`tar -xzf MIKE_Zero_2024_Examples_Update_1.tgz`\
\
`cd $MIKE_TGZ/MIKE_Zero_2024_rhel9_Update_1`\
`sed -i 's/ cp -rp / cp -r /' install.sh`\
`sh install.sh --eula --install-path "$MIKE_HOME" --license-server 127.0.0.1`\
`cd $MIKE_TGZ/MIKE_Zero_2024_Tools_rhel9_Update_1`\
`sed -i 's/ cp -rp / cp -r /' install.sh`\
`sh install.sh --eula --install-path "$MIKE_HOME"`\
`cd $MIKE_TGZ/MIKE_Zero_2024_Examples_Update_1`\
`sed -i 's/ cp -rp / cp -r /' install.sh`\
`sh install.sh --eula --install-path "$MIKE_HOME"`\
\
`module load StdEnv/2023 intel/2023.2.1 intelmpi/2021.9.0`\
`setrpaths.sh --path "$MIKE_HOME/bin"  --add_origin  \`\
`    --add_path="$EBROOTIMPI/mpi/latest/lib/release:$EBROOTIMPI/mpi/latest/lib"`

`</tab>`{=html} `<tab name="MIKE 2023">`{=html}

`export MIKE_TGZ="$HOME/scratch/MIKE_TGZ"`\
`export MIKE_HOME="$HOME/MIKE/2023"`\
\
`cd $MIKE_TGZ`\
`tar -xzf MIKE_Zero_2023_rhel7_22.11.05.tgz`\
`tar -xzf MIKE_Zero_2023_Tools_rhel7_22.11.05.tgz`\
`tar -xzf MIKE_Zero_2023_Examples.tgz`\
\
`cd $MIKE_TGZ/MIKE_Zero_2023_rhel7_22.11.05`\
`sh install.sh --eula --install-path "$MIKE_HOME" --license-server 127.0.0.1`\
`cd $MIKE_TGZ/MIKE_Zero_2023_Tools_rhel7_22.11.05`\
`sh install.sh --eula --install-path "$MIKE_HOME"`\
`cd $MIKE_TGZ/MIKE_Zero_2023_Examples`\
`sh install.sh --eula --install-path "$MIKE_HOME"`\
\
`module load StdEnv/2020  intel/2021.2.0  intelmpi/2021.2.0`\
`setrpaths.sh --path "$MIKE_HOME/bin"  --add_origin  \`\
`    --add_path="$EBROOTIMPI/mpi/latest/lib/release:$EBROOTIMPI/mpi/latest/lib"`

`</tab>`{=html} `<tab name="MIKE 2022">`{=html}

`MIKE_TGZ_DIR="$HOME/MIKE_TGZ"`\
`MIKE_INST_DIR="$HOME/MIKE/2022"`\
\
`cd $MIKE_TGZ_DIR`\
`tar -xzf MIKE_Zero_2022_rhel7_Update_1.tgz `\
`tar -xzf MIKE_Zero_2022_Tools_rhel7_Update_1.tgz`\
`tar -xzf MIKE_Zero_2022_Examples_Update_1.tgz`\
\
`cd $MIKE_TGZ_DIR/MIKE_Zero_2022_rhel7_Update_1`\
`sh install.sh --eula --install-path "$MIKE_INST_DIR" --license-server 127.0.0.1`\
`cd $MIKE_TGZ_DIR/MIKE_Zero_2022_Tools_rhel7_Update_1`\
`sh install.sh --eula --install-path "$MIKE_INST_DIR"`\
`cd $MIKE_TGZ_DIR/MIKE_Zero_2022_Examples_Update_1`\
`sh install.sh --eula --install-path "$MIKE_INST_DIR"`\
\
`module load StdEnv/2020 intel/2020.1.217 intelmpi/2019.7.217`\
`setrpaths.sh --path "$MIKE_INST_DIR/bin"  --add_origin  \`\
`    --add_path="$EBROOTIMPI/intel64/lib/release:$EBROOTIMPI/intel64/lib"`

`</tab>`{=html} `</tabs>`{=html}

### Other versions {#other_versions}

The instructions above assume specific filenames for the installation archives. When installing minor updates released in the same year, the filenames for the archives (e.g. in `tar -xzf MIKE_Zero_2023_rhel7_22.11.05.tgz`), as well as the directory names (e.g. in `cd $MIKE_TGZ/MIKE_Zero_2023_rhel7_22.11.05`) need to be adjusted accordingly. Future major releases of MIKE may use a newer version of Intel MPI, so the above instructions may need to be adapted accordingly. Try a module of the Intel MPI library with a matching Major version (i.e. year).

Essentially the above instructions follow the official installation procedure with the exception that the installation of `MIKE_Zero_*_Prerequisites.tgz` (Intel MPI library) is skipped and a matching module is loaded instead. Furthermore the `setrpaths.sh` script is used to [patch the installed binaries](https://docs.alliancecan.ca/Installing_software_in_your_home_directory#Installing_binary_packages "patch the installed binaries"){.wikilink} to make them compatible with our software stack.

If you run into problems adapting the recipe for newer versions of MIKE, contact our [Technical support](https://docs.alliancecan.ca/Technical_support "Technical support"){.wikilink}.

### Create a module {#create_a_module}

Paste these commands into your terminal to create an environment module for MIKE. Make sure to adjust the version (e.g. \"2025\") to match the version you have installed. Also adjust the version of the `intelmpi` and `intel` modules to match what you had loaded during the installation. After running the commands below, do a fresh login to have the newly created environment module become visible to \"module\" commands or run `module use $HOME/modulefiles`.

`<tabs>`{=html} `<tab name="MIKE 2025">`{=html}

`export MIKE_VERSION=2025`\
`mkdir -p $HOME/modulefiles/mike`\
`cat > $HOME/modulefiles/mike/${MIKE_VERSION}.lua <<EOF`\
`help(`[`Module for MIKE ${MIKE_VERSION} (by DHI group)`](https://docs.alliancecan.ca/Module_for_MIKE_${MIKE_VERSION}_(by_DHI_group) "Module for MIKE ${MIKE_VERSION} (by DHI group)"){.wikilink}`)`\
`local version = "${MIKE_VERSION}"`\
`whatis("Version:".. version)`\
`whatis("Keywords: FEM, Finite Elements, Simulation")`\
`whatis("URL: `[`https://www.mikepoweredbydhi.com/mike-`](https://www.mikepoweredbydhi.com/mike-)`" .. version)`\
`whatis("Description: MIKE is a hydraulic and hydrological modeling software package.")`\
\
`local home = os.getenv("HOME") or "~"`\
`local root = pathJoin( home, "MIKE", version)`\
\
`depends_on("StdEnv/2023", "intel/2023.2.1", "intelmpi/2021.9.0")`\
`setenv("I_MPI_PMI_LIBRARY", "/opt/software/slurm/lib/libpmi2.so")`\
`setenv("SLURM_MPI_TYPE", "pmi2")`\
`setenv("MIKE_HOME", root)`\
`setenv("MIKE_PROGRESS", "STDOUT")`\
`prepend_path( "PATH", pathJoin(root, "bin"))`\
`EOF`

`</tab>`{=html} `<tab name="MIKE 2024">`{=html}

`export MIKE_VERSION=2024`\
`mkdir -p $HOME/modulefiles/mike`\
`cat > $HOME/modulefiles/mike/${MIKE_VERSION}.lua <<EOF`\
`help(`[`Module for MIKE ${MIKE_VERSION} (by DHI group)`](https://docs.alliancecan.ca/Module_for_MIKE_${MIKE_VERSION}_(by_DHI_group) "Module for MIKE ${MIKE_VERSION} (by DHI group)"){.wikilink}`)`\
`local version = "${MIKE_VERSION}"`\
`whatis("Version:".. version)`\
`whatis("Keywords: FEM, Finite Elements, Simulation")`\
`whatis("URL: `[`https://www.mikepoweredbydhi.com/mike-`](https://www.mikepoweredbydhi.com/mike-)`" .. version)`\
`whatis("Description: MIKE is a hydraulic and hydrological modeling software package.")`\
\
`local home = os.getenv("HOME") or "~"`\
`local root = pathJoin( home, "MIKE", version)`\
\
`depends_on("StdEnv/2023", "intel/2023.2.1", "intelmpi/2021.9.0")`\
`setenv("I_MPI_PMI_LIBRARY", "/opt/software/slurm/lib/libpmi2.so")`\
`setenv("SLURM_MPI_TYPE", "pmi2")`\
`setenv("MIKE_HOME", root)`\
`setenv("MIKE_PROGRESS", "STDOUT")`\
`prepend_path( "PATH", pathJoin(root, "bin"))`\
`EOF`

`</tab>`{=html} `<tab name="MIKE 2023">`{=html}

`export MIKE_VERSION=2023`\
`mkdir -p $HOME/modulefiles/mike`\
`cat > $HOME/modulefiles/mike/${MIKE_VERSION}.lua <<EOF`\
`help(`[`Module for MIKE ${MIKE_VERSION} (by DHI group)`](https://docs.alliancecan.ca/Module_for_MIKE_${MIKE_VERSION}_(by_DHI_group) "Module for MIKE ${MIKE_VERSION} (by DHI group)"){.wikilink}`)`\
`local version = "${MIKE_VERSION}"`\
`whatis("Version:".. version)`\
`whatis("Keywords: FEM, Finite Elements, Simulation")`\
`whatis("URL: `[`https://www.mikepoweredbydhi.com/mike-`](https://www.mikepoweredbydhi.com/mike-)`" .. version)`\
`whatis("Description: MIKE is a hydraulic and hydrological modeling software package.")`\
\
`local home = os.getenv("HOME") or "~"`\
`local root = pathJoin( home, "MIKE", version)`\
\
`depends_on("StdEnv/2020", "intel/2021.2.0", "intelmpi/2021.2.0")`\
\
`setenv("I_MPI_PMI_LIBRARY", "/opt/software/slurm/lib/libpmi2.so")`\
`setenv("SLURM_MPI_TYPE", "pmi2")`\
`setenv("MIKE_HOME", root)`\
`setenv("MIKE_PROGRESS", "STDOUT")`\
`prepend_path( "PATH", pathJoin(root, "bin"))`\
`EOF`

`</tab>`{=html} `<tab name="MIKE 2022">`{=html}

`export MIKE_VERSION=2022`\
`mkdir -p $HOME/modulefiles/mike`\
`cat > $HOME/modulefiles/mike/${MIKE_VERSION}.lua <<EOF`\
`help(`[`Module for MIKE ${MIKE_VERSION} (by DHI group)`](https://docs.alliancecan.ca/Module_for_MIKE_${MIKE_VERSION}_(by_DHI_group) "Module for MIKE ${MIKE_VERSION} (by DHI group)"){.wikilink}`)`\
`local version = "${MIKE_VERSION}"`\
`whatis("Version:".. version)`\
`whatis("Keywords: FEM, Finite Elements, Simulation")`\
`whatis("URL: `[`https://www.mikepoweredbydhi.com/mike-`](https://www.mikepoweredbydhi.com/mike-)`" .. version)`\
`whatis("Description: MIKE is a hydraulic and hydrological modeling software package.")`\
\
`local home = os.getenv("HOME") or "~"`\
`local root = pathJoin( home, "MIKE", version)`\
\
`depends_on("StdEnv/2020", "intel/2020.1.217", "intelmpi/2019.7.217") `\
\
`setenv("I_MPI_PMI_LIBRARY", "/opt/software/slurm/lib/libpmi2.so")`\
`setenv("SLURM_MPI_TYPE", "pmi2")`\
`setenv("MIKE_HOME", root)`\
`setenv("MIKE_PROGRESS", "STDOUT")`\
`prepend_path( "PATH", pathJoin(root, "bin"))`\
`EOF`

`</tab>`{=html} `</tabs>`{=html}

Activate this module in each job or login session with:

`<tabs>`{=html} `<tab name="MIKE 2025">`{=html}

`</tab>`{=html} `<tab name="MIKE 2024">`{=html}

`</tab>`{=html} `<tab name="MIKE 2023">`{=html}

`</tab>`{=html} `<tab name="MIKE 2022">`{=html}

`</tab>`{=html} `</tabs>`{=html}

### Configure the license {#configure_the_license}

From MIKE Customer Care you will have instructions like this for configuring your license: internet \--iuseruser@example.com \--ipasswordmy-password}} This normally needs to be done only once whenever you get a new license or license code. The license information will be stored in a file `~/.config/DHI/license/NetLmLcwConfig.xml`.

## Example job script {#example_job_script}

`<tabs>`{=html} `<tab name="MIKE 2025">`{=html} `{{File
|name=job_mike_2025_CPU.sh
|lang="bash"
|contents=
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=00:20:00

module load  StdEnv/2023  intel/2023.2.1  intelmpi/2021.9.0  mike/2025
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

engine="FemEngineHD"
model="my_model.m3fm"

srun $engine $model
}}`{=mediawiki} `</tab>`{=html} `<tab name="MIKE 2024">`{=html} `{{File
|name=job_mike_2024_CPU.sh
|lang="bash"
|contents=
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=00:20:00

module load  StdEnv/2023  intel/2023.2.1  intelmpi/2021.9.0  mike/2024
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

engine="FemEngineHD"
model="my_model.m3fm"

srun $engine $model
}}`{=mediawiki} `</tab>`{=html} `<tab name="MIKE 2023">`{=html} `{{File
|name=job_mike_2023_CPU.sh
|lang="bash"
|contents=
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=00:20:00

module load StdEnv/2020  intel/2021.2.0  intelmpi/2021.2.0  mike/2023
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

engine="FemEngineHD"
model="my_model.m3fm"

srun $engine $model
}}`{=mediawiki} `</tab>`{=html} `<tab name="MIKE 2022">`{=html} `{{File
|name=job_mike_2022_CPU.sh
|lang="bash"
|contents=
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=00:20:00

module load StdEnv/2020  intel/2020.1.217  intelmpi/2019.7.217  mike/2022
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
slurm_hl2hl.py --format MPIHOSTLIST > machinefile.$SLURM_JOBID

engine="FemEngineHD"
model="my_model.m3fm"

mpirun -machinefile machinefile.$SLURM_JOBID $engine $model
}}`{=mediawiki} `</tab>`{=html} `</tabs>`{=html}
