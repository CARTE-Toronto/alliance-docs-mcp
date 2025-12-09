---
title: "Automating VM creation/en"
url: "https://docs.alliancecan.ca/wiki/Automating_VM_creation/en"
category: "General"
last_modified: "2024-07-15T19:47:03Z"
page_id: 740
display_title: "Automating VM creation"
---

Parent page: Cloud

To automate the creation of cloud VMs, volumes, etc. the  OpenStack CLI, Heat, Terraform, or the OpenStack python API can be used. Both the OpenStack CLI and Terraform are command line tools. While Heat is used through the OpenStack web dashboard, horizon. To install and configure settings and software within the VM,  cloud-init is used.

In addition to these tools to create and provision your VMs, you can also gain access to the Compute Canada software stack (CVMFS) that is available on our general purpose computing clusters, within your VM. See the  Enabling CVMFS section below.

==Enabling CVMFS on your VM==
CVMFS is a HTTP-based file system that provides a scalable, reliable, and low maintenance research software distribution service. At the client end, users just need to mount CVMFS and then use the software or libraries directly without worrying about compiling, building, or patching. All the software are pre-compiled for common OS flavors and even modularized so that users can simply load a software as a module.
CVMFS has already been installed on Compute Canada cluster systems such as Cedar, Graham, and Beluga, while on cloud systems users need to enable it by hand, following these cloud instructions: To enable CVMFS on CC Clouds.

For more information please see the Compute Canada CVMFS documentation and CERN CVMFS documentation

==Using cloud-init==
Cloud-init files are used to initialize a particular VM and run within that VM. They can be thought of as a way to automate tasks you would perform at the command line while logged into your VM. They can be used to perform tasks such as updating the operating system, installing and configuring applications, creating files, running commands, and create users and groups. Cloud-init can be used to setup other provisioning tools such as ansible or puppet to continue with the software and VM configuration if desired.

Cloud-init configuration is specified using plain text in the YAML format. To see how to create cloud-init files see the official cloud-init documentation. cloud-init files can be used with the Horizon dashboard (OpenStack's web GUI), Terraform, the CLI, or the Python API. Here we describe how to use a cloud-iinit file with Horizon.

===Specifying a cloud-init File===
# Start as normal when launching an instance, by clicking  under Project->Compute->Instances and specifying your VM's configuration as described in  Launching a VM.
# Before clicking Launch, select the Post-Creation tab and specify your Customization Script Source, in this case a Cloud-init YAML file, by either copying and pasting into a text box (Direct Input method) or uploading from a file from your desktop computer (File method). Older versions of OpenStack, in particular IceHouse, only provide a text box to copy and past your CloudInit file into.
# Once the usual selections for your VM, as described in  Launching a VM, have been made and the Cloud-init YAML file is included, click Launch to create the VM. It may take some time for CloudInit to complete depending on what has been specified in the Cloud-init YAML file.

===Checking Cloud-init Progress===
To see the progress of Cloud-init on a VM, check the console log of the VM by:

# Selecting Project->Compute->Instances in the left hand menu.
# Click on the Instance Name of the VM. This will provide more information about the particular VM.
# Select the Log tab and look for lines containing 'clout-init' for information about the various phases of CloudInit.
# When Cloud-init has finished running the following line will appear near or at the end of the log:

Cloud-init v. 0.7.5 finished at Wed, 22 Jun 2016 17:52:29 +0000. Datasource DataSourceOpenStack [net,ver=2].  Up 44.33 seconds

The log must be refreshed manually by clicking the Go button

==Using Heat Templates==
Heat templates are even more powerful, they can be used to automate tasks performed in the OpenStack dashboard such as creating multiple VMs at once, configuring security groups, creating and configuring networks, and creating and attaching volumes to VMs. Heat templates can be used in conjunction with cloud-init files, once Heat has created the VM it can pass a cloud-init file to that VM to perform setup tasks and even include information about other resources dynamically in the cloud-init files (e.g. floating IPs of other VMs).

As with cloud-init the creation of Heat Orchestration Template (HOT) files is not covered here, instead see the official documentation. HOT files are also written in the YAML format. Heat allows automation of operations performed in the OpenStack dashboard (Horizon) as well as the ability to pass information into the embedded CloudInit files, such as an IP of another server. Before using a Heat template there is usually no need to create any resources in advance. In fact it is often good practice to remove any resources you are not currently using before hand, as using a Heat template consumes resources towards your quota and will fail if it tries to exceed your quota.

To create a stack using a HOT file:

# Select Project->Orchestration->Stacks and click the Launch Stack button to start creating a new stack.
# Provide a HOT file by entering the URL, the File name, or by Direct Input. Here, we will use a HOT file from one of the links in section Available Setups below.
# In the Template Source box, select URL from the drop-down list.
# Paste the selected URL into the Template URL box.
# Click Next to begin setting stack parameters; these vary depending on the template, however all stacks have the following parameters by default:
#*Stack Name; choose a name which is meaningful.
#*Creation Timeout; indicates how long after stack creation before OpenStack will give up trying to create the stack if it hasn't finished; the default value is usually sufficient.
#*Password for user ; sets the password required for later stack changes. This is seldom used as many of the stacks mentioned in the next section are not designed to be updated.
# Click Launch to begin creating your stack.

To graphically see the progress of your stack creation click on the Stack Name and select the Topology tab. Gray nodes indicate that creation is in progress, green nodes have finished being created, and red nodes indicate failures. Once the stack has completed successfully click the Overview tab to see any information that the stack may provide (e.g. a URL to access a service or website).