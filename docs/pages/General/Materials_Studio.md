---
title: "Materials Studio/en"
url: "https://docs.alliancecan.ca/wiki/Materials_Studio/en"
category: "General"
last_modified: "2023-07-19T21:14:01Z"
page_id: 8006
display_title: "Materials Studio"
---

`<languages />`{=html}

The Alliance does not have permission to install Materials Studio centrally on all clusters. If you have a license, follow these instructions to install the application in your account. Please note that the current instructions are only valid for older standard software environments, so before beginning you will need to use a command like `module load StdEnv/2016.4` if you are using the default 2020 [standard software environment](https://docs.alliancecan.ca/Standard_software_environments "standard software environment"){.wikilink}.

# Installing Materials Studio 2020 {#installing_materials_studio_2020}

If you have access to Materials Studio 2020, you will need two things to proceed. First, you must have the archive file that contains the installer; this file should be named `BIOVIA_2020.MaterialsStudio2020.tar`. Second, you must have the IP address (or DNS name) and the port of an already configured license server to which you will connect.

Once you have these, upload the `BIOVIA_2020.MaterialsStudio2020.tar` file to your /home folder on the cluster you intend to use. Then, run the commands `<port>`{=html}@`<server>`{=html}}} and \$HOME}}

Once this command has completed, log out of the cluster and log back in. You should then be able to load the module with

In order to be able to access the license server from the compute nodes, you will need to [contact technical support](https://docs.alliancecan.ca/Technical_support "contact technical support"){.wikilink} so that we can configure our firewall(s) to allow the software to connect to your licence server.

# Installing Materials Studio 2018 {#installing_materials_studio_2018}

If you have access to Materials Studio 2018, you will need two things to proceed. First, you must have the archive file that contains the installer; this file should be named `MaterialsStudio2018.tgz`. Second, you must have the IP address (or DNS name) and the port of an already configured license server to which you will connect.

Once you have these, upload the `MaterialsStudio2018.tgz` file to your /home folder on the cluster you intend to use. Then, run the commands `<port>`{=html}@`<server>`{=html}}} and \$HOME}}

Once this command has completed, log out of the cluster and log back in. You should then be able to load the module with

In order to be able to access the license server from the compute nodes, you will need to [contact technical support](https://docs.alliancecan.ca/Technical_support "contact technical support"){.wikilink} so that we can configure our firewall(s) to allow the software to connect to your licence server.

## Team installation {#team_installation}

If you are a PI holding the Materials Studio licence, you can install Materials Studio once for all your group members. Since normally team work is stored in the `/project` space, determine which project directory you want to use. Suppose it is `~/projects/A_DIRECTORY`, then you will need to know these two values:

1\. Determine the actual path of A_DIRECTORY as follows: \$(readlink -f \~/projects/A_DIRECTORY)\|echo \$PI_PROJECT_DIR}} 2. Determine the group of A_DIRECTORY as follows: \$(stat -c%G \$PI_PROJECT_DIR)\|echo \$PI_GROUP}}

With these values known, install Materials Studio.

1.  Change the default group to your team\'s `def-` group, e.g.,
2.  Open the permissions of your project directory so your team can access it, e.g.,
3.  Create an install directory within /project, e.g.,
4.  Install the software, e.g., `<port>`{=html}@`<server>`{=html} eb MaterialsStudio-2018-dummy-dummy.eb \--installpath\$PI_PROJECT_DIR/MatStudio2018 \--sourcepath\$HOME}}

Before the software can be run:

1.  Run this command.
    - Your team members may wish to add this to their `~/.bashrc` file.
2.  Load the materialsstudio module, i.e.,

`<b>`{=html}NOTE:`</b>`{=html} Be sure to always replace variables PI_GROUP and PI_PROJECT_DIR with their appropriate values.

# Examples of Slurm job submission scripts {#examples_of_slurm_job_submission_scripts}

The following examples assume that you have installed Materials Studio 2018 according to the above instructions.

Below is an example of a Slurm job script that relies on Materials Studio\'s RunCASTEP.sh command:

# Installing earlier versions of Materials Studio {#installing_earlier_versions_of_materials_studio}

If you require an earlier version of Materials Studio than 2018, you will need to install in into an [Apptainer](https://docs.alliancecan.ca/Apptainer "Apptainer"){.wikilink} container. This involves

1.  creating an Apptainer container with a compatible distribution of Linux installed in it;
2.  installing Materials Studio into that container;
3.  uploading the Apptainer container to your account and using it there.
    - NOTE: In order to be able to access the license server from the compute nodes, you will need to [contact technical support](https://docs.alliancecan.ca/Technical_support "contact technical support"){.wikilink} so that we can configure our firewall(s) to allow the software to connect to your license server.

Please be aware that you might be restricted to whole-node (single-node) jobs as the version of MPI inside the container might not be able to be used across nodes.
