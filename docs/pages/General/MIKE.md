---
title: "MIKE/en"
url: "https://docs.alliancecan.ca/wiki/MIKE/en"
category: "General"
last_modified: "2025-09-03T12:24:42Z"
page_id: 21986
display_title: "MIKE"
---

MIKE powered by DHI is a a hydraulic and hydrological modeling software package.

== License requirements ==
MIKE is a commercial product and each user needs to supply their own license.

In order for you to use it on our HPC clusters, you will need to contact MIKE Customer Care
at: mike@dhigroup.com and confirm that you have
* an internet license, and
* a download link for the Linux version of MIKE.

== Installation ==

You need to download the installation archives for Linux.

The following instructions assume that the installation archives are in one Zip-file (MIKE 2025 and newer) or three  *.tgz files (MIKE 2024 and older):

* MIKE_Zero_2025_rhel9.zip

* MIKE_Zero_2024_rhel9_Update_1.tgz
* MIKE_Zero_2024_Tools_rhel9_Update_1.tgz
* MIKE_Zero_2024_Examples_Update_1.tgz

* MIKE_Zero_2023_rhel7_22.11.05.tgz
* MIKE_Zero_2023_Tools_rhel7_22.11.05.tgz
* MIKE_Zero_2023_Examples.tgz

* MIKE_Zero_2022_rhel7_Update_1.tgz
* MIKE_Zero_2022_Tools_rhel7_Update_1.tgz
* MIKE_Zero_2022_Examples_Update_1.tgz

1. Create a directory ~/scratch/MIKE_TGZ and upload the installation archive(s) to that location.

2. MIKE was compiled with the Intel MPI library, therefore you must load a matching intelmpi module.

 module load StdEnv/2023 intel/2023.2.1 intelmpi/2021.9.0

 module load StdEnv/2020  intel/2021.2.0  intelmpi/2021.2.0

 module load StdEnv/2020  intel/2020.1.217  intelmpi/2019.7.217

3. Run the following commands depending on the version of MIKE.
They will extract the archives, run the `install.sh` installation scripts for each component
and then Patch the binaries so that they can find
the dynamic libraries of Intel MPI.

 export MIKE_TGZ="$HOME/scratch/MIKE_TGZ"
 export MIKE_HOME="$HOME/MIKE/2025"

 cd $MIKE_TGZ
 unzip -j  MIKE_Zero_2025_rhel9.zip
 tar -xzf MIKE_Common_2025_rhel9.tgz
 tar -xzf MIKE_Zero_2025_rhel9.tgz
 tar -xzf MIKE_Zero_2025_Tools_rhel9.tgz
 tar -xzf MIKE_Zero_2025_Examples.tgz

 cd $MIKE_TGZ/MIKE_Common_2025_rhel9
 sed -i 's/ cp -rp / cp -r /' install.sh
 sh install.sh --eula --install-path "$MIKE_HOME" --license-server 127.0.0.1
 cd $MIKE_TGZ/MIKE_Zero_2025_rhel9
 sed -i 's/ cp -rp / cp -r /' install.sh
 sh install.sh --eula --install-path "$MIKE_HOME"
 cd $MIKE_TGZ/MIKE_Zero_2025_Tools_rhel9
 sed -i 's/ cp -rp / cp -r /' install.sh
 sh install.sh --eula --install-path "$MIKE_HOME"
 cd $MIKE_TGZ/MIKE_Zero_2025_Examples
 sed -i 's/ cp -rp / cp -r /' install.sh
 sh install.sh --eula --install-path "$MIKE_HOME"

 module load StdEnv/2023 intel/2023.2.1 intelmpi/2021.9.0
 setrpaths.sh --path "$MIKE_HOME/bin"  --add_origin  \
     --add_path="$EBROOTIMPI/mpi/latest/lib/release:$EBROOTIMPI/mpi/latest/lib"

 export MIKE_TGZ="$HOME/scratch/MIKE_TGZ"
 export MIKE_HOME="$HOME/MIKE/2024"

 cd $MIKE_TGZ
 tar -xzf MIKE_Zero_2024_rhel9_Update_1.tgz
 tar -xzf MIKE_Zero_2024_Tools_rhel9_Update_1.tgz
 tar -xzf MIKE_Zero_2024_Examples_Update_1.tgz

 cd $MIKE_TGZ/MIKE_Zero_2024_rhel9_Update_1
 sed -i 's/ cp -rp / cp -r /' install.sh
 sh install.sh --eula --install-path "$MIKE_HOME" --license-server 127.0.0.1
 cd $MIKE_TGZ/MIKE_Zero_2024_Tools_rhel9_Update_1
 sed -i 's/ cp -rp / cp -r /' install.sh
 sh install.sh --eula --install-path "$MIKE_HOME"
 cd $MIKE_TGZ/MIKE_Zero_2024_Examples_Update_1
 sed -i 's/ cp -rp / cp -r /' install.sh
 sh install.sh --eula --install-path "$MIKE_HOME"

 module load StdEnv/2023 intel/2023.2.1 intelmpi/2021.9.0
 setrpaths.sh --path "$MIKE_HOME/bin"  --add_origin  \
     --add_path="$EBROOTIMPI/mpi/latest/lib/release:$EBROOTIMPI/mpi/latest/lib"

 export MIKE_TGZ="$HOME/scratch/MIKE_TGZ"
 export MIKE_HOME="$HOME/MIKE/2023"

 cd $MIKE_TGZ
 tar -xzf MIKE_Zero_2023_rhel7_22.11.05.tgz
 tar -xzf MIKE_Zero_2023_Tools_rhel7_22.11.05.tgz
 tar -xzf MIKE_Zero_2023_Examples.tgz

 cd $MIKE_TGZ/MIKE_Zero_2023_rhel7_22.11.05
 sh install.sh --eula --install-path "$MIKE_HOME" --license-server 127.0.0.1
 cd $MIKE_TGZ/MIKE_Zero_2023_Tools_rhel7_22.11.05
 sh install.sh --eula --install-path "$MIKE_HOME"
 cd $MIKE_TGZ/MIKE_Zero_2023_Examples
 sh install.sh --eula --install-path "$MIKE_HOME"

 module load StdEnv/2020  intel/2021.2.0  intelmpi/2021.2.0
 setrpaths.sh --path "$MIKE_HOME/bin"  --add_origin  \
     --add_path="$EBROOTIMPI/mpi/latest/lib/release:$EBROOTIMPI/mpi/latest/lib"

 MIKE_TGZ_DIR="$HOME/MIKE_TGZ"
 MIKE_INST_DIR="$HOME/MIKE/2022"

 cd $MIKE_TGZ_DIR
 tar -xzf MIKE_Zero_2022_rhel7_Update_1.tgz
 tar -xzf MIKE_Zero_2022_Tools_rhel7_Update_1.tgz
 tar -xzf MIKE_Zero_2022_Examples_Update_1.tgz

 cd $MIKE_TGZ_DIR/MIKE_Zero_2022_rhel7_Update_1
 sh install.sh --eula --install-path "$MIKE_INST_DIR" --license-server 127.0.0.1
 cd $MIKE_TGZ_DIR/MIKE_Zero_2022_Tools_rhel7_Update_1
 sh install.sh --eula --install-path "$MIKE_INST_DIR"
 cd $MIKE_TGZ_DIR/MIKE_Zero_2022_Examples_Update_1
 sh install.sh --eula --install-path "$MIKE_INST_DIR"

 module load StdEnv/2020 intel/2020.1.217 intelmpi/2019.7.217
 setrpaths.sh --path "$MIKE_INST_DIR/bin"  --add_origin  \
     --add_path="$EBROOTIMPI/intel64/lib/release:$EBROOTIMPI/intel64/lib"

=== Other versions ===

The instructions above assume specific filenames for the installation archives.  When installing minor updates released in the same year,
the filenames for the archives (e.g. in tar -xzf MIKE_Zero_2023_rhel7_22.11.05.tgz),
as well as the directory names (e.g. in cd $MIKE_TGZ/MIKE_Zero_2023_rhel7_22.11.05) need to be adjusted accordingly.
Future major releases of MIKE may use a newer version of Intel MPI, so the above instructions may need to be adapted
accordingly. Try a module of the Intel MPI library with a matching Major version (i.e. year).

Essentially the above instructions follow the official installation procedure with the exception that the installation of MIKE_Zero_*_Prerequisites.tgz (Intel MPI library) is skipped and a matching module is loaded instead. Furthermore the setrpaths.sh script is used to patch the installed binaries to make them compatible with our software stack.

If you run into problems adapting the recipe for newer versions of MIKE, contact our Technical support.

=== Create a module ===

Paste these commands into your terminal to create an environment module for MIKE.
Make sure to adjust the version (e.g. "2025") to match the version you have installed.
Also adjust the version of the intelmpi and intel modules
to match what you had loaded during the installation.
After running the commands below, do a fresh login to have the newly created environment module become visible to "module" commands or run module use $HOME/modulefiles.

 export MIKE_VERSION=2025
 mkdir -p $HOME/modulefiles/mike
 cat > $HOME/modulefiles/mike/${MIKE_VERSION}.lua <

 export MIKE_VERSION=2024
 mkdir -p $HOME/modulefiles/mike
 cat > $HOME/modulefiles/mike/${MIKE_VERSION}.lua <

 export MIKE_VERSION=2023
 mkdir -p $HOME/modulefiles/mike
 cat > $HOME/modulefiles/mike/${MIKE_VERSION}.lua <

 export MIKE_VERSION=2022
 mkdir -p $HOME/modulefiles/mike
 cat > $HOME/modulefiles/mike/${MIKE_VERSION}.lua <

Activate this module in each job or login session with:

=== Configure the license ===

From MIKE Customer Care you will have instructions like this for configuring your license:
internet --iuseruser@example.com --ipasswordmy-password}}
This normally needs to be done only once whenever you get a new license or license code.
The license information will be stored in a file ~/.config/DHI/license/NetLmLcwConfig.xml.

== Example job script ==