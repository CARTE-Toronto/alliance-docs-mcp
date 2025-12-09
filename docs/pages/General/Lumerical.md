---
title: "Lumerical/en"
url: "https://docs.alliancecan.ca/wiki/Lumerical/en"
category: "General"
last_modified: "2022-12-12T15:34:23Z"
page_id: 6301
display_title: "Lumerical"
---

Lumerical is a suite of applications for modelling nanophotonic devices, which includes FDTD Solutions.

= Installation =
FDTD Solutions is now available as part of the Lumerical package.
Compute Canada does not have a central installation of the Lumerical suite or FDTD Solutions. However, if you are licensed to use the software, you can install it following the instructions below.

If you have downloaded whole Lumerical suite (e.g. filename: Lumerical-2020a-r1-d316eeda68.tar.gz), follow the instructions in sections "Installing Lumerical" and "Using the Lumerical module".
If you have downloaded FDTD Solutions on it's own (e.g. filename: FDTD_Solutions-8.19.1438.tar.gz), follow the instructions in sections "Installing FDTD Solutions" and "Using the fdtd_solutions module".

== Installing Lumerical ==
=== In case the installer release matches that of the recipe ===
To install the Lumerical suite run the command  --disable-enforce-checksums}}
where path is the path to the folder containing the .tar.gz file to install Lumerical on Linux.

=== In case the installer release does not match that of the recipe ===
With a different 2020a release than 2020a-r1-d316eeda68, run
 --sourcepath --disable-enforce-checksums}}
For example, if Lumerical-2020a-r1-d316eeda68.eb.tar.gz is downloaded in $HOME/scratch, the following command will install Lumerical within your $HOME/.local folder.
2020a-r6-aabbccdd --sourcepath$HOME/scratch --disable-enforce-checksums}}

It is important that the version of the installation recipe (year plus 'a' or 'b') needs to exactly match that of the installer.
If either the letter or the year changes (e.g. from 2020a to 2020b), we will need to adapt the installation script to the new version.

As of April 1st, 2020 we have the following installation recipes available:

Installation recipe                   	Intended for Installer                    	Compatible with Installers
Lumerical-2019b-r6-1db3676.eb         	Lumerical-2019b-r6-1db3676.tar.gz         	Lumerical-2019b-*.tar.gz
Lumerical-2020a-r1-d316eeda68.eb      	Lumerical-2020a-r1-d316eeda68.tar.gz      	Lumerical-2020a-*.tar.gz
Lumerical-2021-R2.5-2885-27742aa972.eb	Lumerical-2021-R2.5-2885-27742aa972.tar.gz	Lumerical-2021-*.tar.gz
Lumerical-2022-R1.3-3016-2c0580a.eb   	Lumerical-2022-R1.3-3016-2c0580a.tar.gz   	Lumerical-2022-*.tar.gz

If this does not work, please contact our Technical support and we will adapt an installation recipe for your version.

Once installed, you will need to log out and back into the server. To load the Lumerical module, use

=== Configuring your own license file ===

The Lumerical module will look for the file $HOME/.licenses/lumerical.lic to determine how to contact the license server.
Create the file with the following content, adjusting 27011@license01.example.com to the port and hostname of your license server.

copy the content below to $HOME/.licenses/lumerical.lic
 setenv("LUMERICAL_LICENSE_FILE", "27011@license01.example.com")

== Installing FDTD Solutions ==
To install FDTD Solutions, run the command  --disable-enforce-checksums}}
where path is the path to the folder containing the .tar.gz file to install FDTD Solutions on Linux.

With a version other than 8.19.1438, run
 --sourcepath --disable-enforce-checksums}}
For example, if FDTD_Solutions-8.19.1466.tar is downloaded in $HOME/Downloads, the following command will install FDTD Solution within your $HOME/.local folder.
8.19.1466 --sourcepath$HOME/Downloads --disable-enforce-checksums}}

If this does not work, please contact our Technical support and we will adapt an installation script for your version.

Once installed, you will need to log out and back into the server. To load the FDTD module, use

You will also need to set up your installation to use your license server. Start the software first on a login node; it should ask you for information about the license server. You will only need to do this once.

= Using the software =
The main difference between the modules fdtd_solutions and lumerical, beside the fact that the Lumerical module contains additional tools, is that the environment variable that contains the install location is named EBROOTFDTD_SOLUTIONS and EBROOTLUMERICAL respectively.  This means scripts written for one module should be adjusted for the other by replacing the name of the module in the module load ... line and replacing EBROOTFDTD_SOLUTIONS with EBROOTLUMERICAL or vice versa.

== Using the Lumerical module ==

The MPI implementation provided by Lumerical is not tightly coupled with our scheduler. Because of this, you should use options --ntasks-per-node=1 and --cpus-per-task=32 when submitting a job.

Your submission script should look like the following example, where two nodes are requested for 30 minutes. You can adjust the time limit and the node count to fit your needs.

== Using the fdtd_solutions module ==
The MPI implementation provided by FDTD is not tightly coupled with our scheduler. Because of this, you should use options --ntasks-per-node=1 and --cpus-per-task=32 when submitting a job.

Your submission script should look like the following example, where two nodes are requested for one hour. You can adjust the time limit and the node count to fit your needs.

== Templates ==

Note: This section is intended for use with the "fdtd_solutions" module and has not been adapted for "lumerical".

If you are performing a lot of simulations, you may find it inefficient to edit the job submission script for each simulation. You can use template submission scripts to improve this.

For example:
* Create directory $HOME/bin and put the main script fdtd-run.sh (see below) there.
* Create directory $HOME/bin/templates and put the job submission template script fdtd-mpi-template.sh and process template script fdtd-process-template.sh there.

fdtd-mpi-template.sh is basically a shell of the fdtd_solutions.sh script shown above and fdtd-process-template.sh determines the computing resources you need.

To submit a job, run

This will use the 32 cores on a single standard node. If you want to use more cores, request multiple nodes like so:

 sed 's/^.*=//'`

#Total memory required
TOTALMEM=$(( ESTMEM * MEMORY_SAFETY / 100 ))

#Memory required per process
PROCMEM=$((TOTALMEM / PROCS))
if [ "$PROCMEM" -lt "$MEMORY_MIN" ]; then
    PROCMEM=$MEMORY_MIN
fi

#Gridpoints
GRIDPTS=`grep gridpoints $1.tmp  sed 's/^.*=//'`

#Timesteps
TIMESTEPS=`grep time_steps $1.tmp  sed 's/^.*=//'`

#Estimated time
TIME=$(( GRIDPTS * TIMESTEPS / PROCS / RATE / 10000000 ))
if [ "$TIME" -lt "$TIME_MIN" ]; then
    TIME=$TIME_MIN
fi

HOUR=$((TIME / 3600))
MINSEC=$((TIME - HOUR * 3600))
MIN=$((MINSEC / 60))
SEC=$((MINSEC - MIN * 60))

echo $TOTALMEM

#The replacements
sed -e "s##$TOTALMEM#g" \
    -e "s##$PROCMEM#g" \
    -e "s##$HOUR#g" \
    -e "s##$MIN#g" \
    -e "s##$SEC#g" \
    -e "s##$PROCS#g" \
    -e "s##$DIRFSP#g" \
    -e "s##$FILENAME#g" \
    $2
}}