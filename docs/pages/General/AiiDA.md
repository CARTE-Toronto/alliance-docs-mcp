---
title: "AiiDA"
url: "https://docs.alliancecan.ca/wiki/AiiDA"
category: "General"
last_modified: "2025-02-18T18:45:42Z"
page_id: 25587
display_title: "AiiDA"
---

# General

[AiiDA](https://www.aiida.net/) is an open-source Python tool to help researchers with automating and reproducing the complex workflows associated with modern computational science. Numerous plugins exist to integrate AiiDA with common computational packages, particularly in the field of computational chemistry (see list of available [plugins](https://aiidateam.github.io/aiida-registry/) ).

Typically AiiDA will run on user\'s personal computer, a lab workstation, or a virtual machine in the cloud. From there the user can have AiiDA submit jobs to clusters.

To do this, AiiDA must be able to make many SSH connections to the clusters autonomously. This becomes a problem on clusters which require interactive [Multifactor authentication (MFA)](https://docs.alliancecan.ca/Multifactor_authentication "Multifactor authentication (MFA)"){.wikilink}. The implementation of MFA support in AiiDA is still under development.

# [Automation in the context of multifactor authentication](https://docs.alliancecan.ca/Automation_in_the_context_of_multifactor_authentication "Automation in the context of multifactor authentication"){.wikilink} {#automation_in_the_context_of_multifactor_authentication}

To allow AiiDA and other similar software to connect to Alliance clusters, we have set up so-called [automation nodes](https://docs.alliancecan.ca/Automation_in_the_context_of_multifactor_authentication#Automation_nodes_for_each_cluster "automation nodes"){.wikilink} on each cluster. These allow SSH connections without MFA, subject to restrictions.

# Instructions on using AiiDA with Alliance clusters automation nodes {#instructions_on_using_aiida_with_alliance_clusters_automation_nodes}

These instructions assume the reader is already familiar with how AiiDA works.

## Obtain access to automation nodes {#obtain_access_to_automation_nodes}

Users need to submit a ticket requesting access to automation nodes, explaining why it is needed.

## Set up SSH key {#set_up_ssh_key}

The first step is to [upload the public SSH key](https://docs.alliancecan.ca/SSH_Keys#Installing_your_key "upload the public SSH key"){.wikilink} to enable access to Alliance clusters. [Generate](https://docs.alliancecan.ca/SSH_Keys#Generating_an_SSH_Key "Generate"){.wikilink} this key on the machine which will be running AiiDA.

The uploaded key needs to specify the IP address that will be used to connect, as well as the working directory that AiiDA will be using (specified during AiiDA \"computer setup\" stage). If the IP of the machine running AiiDA changes, the user will need to upload a key with updated IP address.

It also needs to specify the location of the aiida_commands.sh script that will be used to manage the connection (so change someusername to the actual username). The key given here is an example only, you should change it to the public key that you generated.

`restrict,from="206.158.1.23",command="/home/someusername/aiida_commands.sh /project/6000000/aiida_work_dir" ssh-ed25519 AAAAC3NzbC1lZKI1NTE5AABBIHyIW8dNHKpNae4jKjtIJW2LIagCPfHiT8yWSj/LXEvV`

Make sure you have no extra unnecessary spaces in your uploaded ssh key. The format must be exactly as given here, but you need to change /project/6000000/aiida_work_dir to the actual directory where you want to put your aiida files.

## Set up automation script {#set_up_automation_script}

The above key will run every SSH command coming in with that key through the aiida_commands.sh script. The user should download this script to the desired Alliance cluster, by running on that cluster:

`wget `[`https://raw.githubusercontent.com/ComputeCanada/software-stack-custom/main/bin/computecanada/allowed_commands/aiida_commands.sh`](https://raw.githubusercontent.com/ComputeCanada/software-stack-custom/main/bin/computecanada/allowed_commands/aiida_commands.sh)

The location should be the one specified in the SSH key.

The file should be given execute permission, and should be made non-writeable for added security.

`chmod u+x aiida_commands.sh`\
`chmod u-w aiida_commands.sh`

This file should not be modified without consultation with Alliance staff.

## Set up computer and code to use in AiiDA {#set_up_computer_and_code_to_use_in_aiida}

In AiiDA, set up a \"computer\" to use the desired cluster, using one of the [ cluster automation nodes](https://docs.alliancecan.ca/Automation_in_the_context_of_multifactor_authentication#Automation_nodes_for_each_cluster " cluster automation nodes"){.wikilink} as hostname. Specify the SSH key defined above as the one to be used.

## Setup on cluster {#setup_on_cluster}

You may need to install the software you wish to run on the cluster. Instructions will differ depending on which computational package you wish to use. You may need to specify the location on the software when you create AiiDA \"code\".

Make sure the working directory specified when you created the \"computer\" in AiiDA to be used for connecting to the cluster exists.

Specify the default account for the jobs to run under, by adding to your .bashrc lines like (change someuser to the actual user name that matches the account you are running under)

`export SLURM_ACCOUNT=def-someuser`\
`export SBATCH_ACCOUNT=$SLURM_ACCOUNT`

You may also need to add this line to your .bashrc in order to get the script working.

`source /etc/profile`

## Debugging

The script has been tested for a basic AiiDA job, but it may run into trouble for more complicated workflows. Please consult with Alliance staff if you run into any difficulties.
