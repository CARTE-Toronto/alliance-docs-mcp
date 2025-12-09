---
title: "FreeSurfer/en"
url: "https://docs.alliancecan.ca/wiki/FreeSurfer/en"
category: "General"
last_modified: "2022-07-08T22:19:58Z"
page_id: 14372
display_title: "FreeSurfer"
---

=Introduction=
FreeSurfer is a set of tools for the analysis and visualization of structural and functional brain imaging data. FreeSurfer contains a fully automatic structural imaging stream for processing cross sectional and longitudinal data.

=FreeSurfer 5.3 as a global module=
In our software stack, you may load the freesurfer/5.3.0 module.

FreeSurfer comes up with a script FreeSurferEnv.sh that should be sourced to correctly set up environment variables such as PATH and PERL5LIB:

=FreeSurfer 6.0 and newer versions=
Due to a change in the license terms, we no longer install the code as a central module. If needed, please install it in your /home directory or in your /project space with EasyBuild. Please follow the instructions below and if needed, contact the technical support for assistance.

==Download the software==
Select a version (6.0.0 or newer) in the download repository and download the corresponding freesurfer-Linux*vX.Y.Z.tar.gz archive on your favorite cluster.

==Installation in your /home directory with EasyBuild==
The following procedure will install FreeSurfer 6.0.0 in /home/$USER/.local/easybuild/software/2020/Core/freesurfer/6.0.0/. The installation requires some memory and due to the restrictions of memory stack size on the login nodes on our clusters, the installation may fail because of the memory. To overcome this issue, you may need to use an interactive job by asking for enough memory (8 GB or so) to install the code.

# Go to the folder that contains the freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.0.tar.gz archive.
# Unload all modules with module purge.
# Install with EasyBuild using eb FreeSurfer-6.0.0-centos6_x86_64.eb --disable-enforce-checksums.
# Register for the FreeSurfer license key .
# Your user license will have to go in

Use nano or any other text editor of your choice and create a file /home/$USER/.license and add the license text (example):

name.name@university.ca
12345
*A1BCdEfGHiJK
ABCd0EFgHijKl

To load the private module:
module load freesurfer/6.0.0

As of August 2020, we were supporting up to version 6.0.1. You can check for newer versions here.

==EasyBuild recipes==
You can check the EasyBuild recipes for FreeSurfer online on GitHub or via a command line ,eb -S FreeSurfer, from any of our clusters. If the version you are looking for is not listed, you may try to install the program with the option --try-software-version=. If that did not work, please contact the technical support for help.

==Installation in a shared folder==
Using EasyBuild, it is possible to install the program in a shared location (like /project) and make the code available for any other member of the group. The following will install FreeSurfer under the directory /home/$USER/projects/def-someuser/$USER/software and the module under the user's directory /home/$USER/.local/easybuild/modules/2020/Core/freesurfer.

newgrp def-someuser
installdir=/home/$USER/projects/def-someuser/$USER
moduledir=/home/$USER/.local/easybuild/modules/2020
pathtosrc=/home/$USER/software
eb FreeSurfer-6.0.1-centos6_x86_64.eb --installpath-modules=${moduledir} --prefix=${installdir} --sourcepath=${pathtosrc}

If it complains about checksums, add the option --disable-enforce-checksums to the eb command.

To make the program accessible for all members of the group, two more steps are required:

* You need to give all members of your group read and exec access to the installation directory /home/$USER/projects/def-someuser/$USER. To see how to give them access to this directory, please read Changing the permissions of existing files.
* Each member of the group will need to put the module file in their own /home directories. The module file 6.0.1.lua is located under the directory:

/home/$USER/.local/easybuild/modules/2020/Core/freesurfer/

Each member of the group will need to create the directory /home/$USER/.local/easybuild/modules/2020/Core/freesurfer where they will put the file 6.0.1.lua:

The above will set the module (only the module file that points to the installation directory under /project) in their own directory.

The module can be loaded from their own accounts using:

==Hippocampus and brainstem processing==
To perform processing of the hippocampus and brainstem, download and install MATLAB runtime 2012b from the FreeSurfer website:

module load freesurfer/6.0.0
cd $FREESURFER_HOME
curl "http://surfer.nmr.mgh.harvard.edu/fswiki/MatlabRuntime?action=AttachFile&do=get⌖=runtime2012bLinux.tar.gz" -o "matlab_runtime2012bLinux.tar.gz"
tar xvf matlab_runtime2012bLinux.tar.gz

==Example of working batch script for FreeSurfer version >= 6.0.0==

==Examples of required walltimes==

* recon-all -all : #SBATCH --time=08:00:00
* recon-all -qcache : #SBATCH --time=00:20:00
* recon-all -base -tp1 -tp2 : #SBATCH --time=10:00:00
* recon-all -long subjid -base base : #SBATCH --time=10:00:00
* recon-all -hippocampal-subfields-T1 : #SBATCH --time=00:40:00
* recon-all -brainstem-structures: #SBATCH --time=00:30:00