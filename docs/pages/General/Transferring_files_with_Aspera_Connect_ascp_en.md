---
title: "Transferring files with Aspera Connect/ascp/en"
url: "https://docs.alliancecan.ca/wiki/Transferring_files_with_Aspera_Connect/ascp/en"
category: "General"
last_modified: "2024-05-31T14:47:01Z"
page_id: 23587
display_title: "Transferring files with Aspera Connect/ascp"
---

`<languages />`{=html}

ascp is a software used for transferring data to and from Aspera transfer servers where you have license credentials. A common use case is uploading a dataset to a data repository.

Many transfer servers will require an up-to-date version of the software (now called Aspera Connect). Due to changes in the software, you may need to install this updated version locally. Please see below for details.

# ascp 3.5.4 {#ascp_3.5.4}

This software is available as a module in older software stacks.

If the server you are trying to transfer data to is incompatible with this older version, you may get an error such as

    Error with Aspera:

    ascp: failed to authenticate, exiting.

    Session Stop  (Error: failed to authenticate)

# Aspera Connect/ascp 4+ {#aspera_connectascp_4}

In order to use Aspera Connect (formerly ascp) you will need to [ install it in one of your local directories](https://docs.alliancecan.ca/Installing_software_in_your_home_directory " install it in one of your local directories"){.wikilink}.

1\. Copy the link for the latest Linux version of the software from the [Aspera Connect website](https://www.ibm.com/aspera/connect).

2\. Go to the directory where you want to install the software, e.g. your /home directory.

3\. Download the software in this directory with wget.

`wget link-i-copied-here`

4\. Extract the software from the archive.

`tar -zxf ibm-aspera-connect_some_version_linux.tar.gz`

5\. Run the setup script.

`bash ibm-aspera-connect_some_version_linux.sh`

5a. Make the library files executable:

`chmod u+x ~/.aspera/connect/plugins/*/*.so ~/.aspera/connect/lib/*`

6\. Run the setrpaths script.

`setrpaths.sh --path $HOME/.aspera`

7\. (Optional) add the ascp binaries to your PATH:

`export PATH=~/.aspera/connect/bin:$PATH`
