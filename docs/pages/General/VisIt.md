---
title: "VisIt/en"
url: "https://docs.alliancecan.ca/wiki/VisIt/en"
category: "General"
last_modified: "2025-08-31T14:16:39Z"
page_id: 10559
display_title: "VisIt"
---

= Client-server visualization =

VisIt requires the same major version on the local client and the remote host; this prevents incompatibility that typically shows as a failed handshake when establishing the client-server connection.

Please use the tabs below to select the remote system.

== Client-server visualization on Fir, Nibi and Rorqual==

We have several versions of VisIt installed on the clusters. You can see them by running module spider visit. To use remote VisIt in client-server mode, on your laptop you need the matching major version, either 2.12.x or 2.13.x or 3.2.x. Before starting VisIt, download the host profile XML file host_fir.xml -- this config file is for connecting to VisIt 3.2.1.

* On Linux/Mac, copy the file to ~/.visit/hosts/.
* On Windows, copy the file to My Documents\VisIt 3.2.1\hosts or similar directory.

Start VisIt on your laptop; in the main menu, under Options -> Host Profiles, you should see the fir host profile. If you want to do remote rendering on Nibi instead, set

 Host nickname = nibi
 Remote host name = nibi.alliancecan.ca

For both Fir and Nibi, set your CCDB username.

 Username = yourOwwUserName

With the exception of your username, your settings should be similar to the ones shown below:

In the same setup window, click on the Launch Profiles tab. You should see two profiles (login and slurm):

* The login profile is for running the engine on a login node, which we do not recommend for heavy visualizations.
* The slurm profile is for running the engine inside an interactive job on a compute node. If you are planning to do this, select the slurm profile and then click on the Parallel tab and below it on the Advanced tab. In the Launcher arguments field, replace --account=def-someuser by your default allocation, as shown below.
* On Fir only, add the flag "--chdir=/scratch/username", specifying your actual username, to launch your job from your scratch directory.

Save the settings with Options -> Save Settings and then exit VisIt on your laptop for settings to take effect.

If your account has multifactor authentication enabled, you will need to configure your SSH client with ControlMaster and make sure to use the full host name for Host, e.g.

Host fir.alliancecan.ca
    HostName fir.alliancecan.ca
    User
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlMaster auto
    ControlPersist 10m

Next, log into the cluster in a terminal window. Finally, relaunch VisIt on your computer, start the file-open dialogue and change the local host to fir (or nibi). Hopefully, the connection is established, the remote VisIt Component Launcher gets started on the cluster's login node, and you should be able to see the cluster's filesystem, navigate to your file and select it. You will be prompted to select either the login (rendering on the login node) or slurm (rendering inside an interactive Slurm job on a compute node) profile. If you select the slurm profile, you will need to specify the number of nodes and processors and the maximum time limit:

Click on OK and wait for VisIt's engine to start. If you selected rendering on a compute node, it may take some time for your job to get started. Once your dataset appears in the Active source, the VisIt's engine is running and you can proceed with creating and drawing your plot.

== Client-server visualization on Trillium==

=== Host configuration ===

For VisIt to connect to the Trillium cluster, you need to configure your host along one of the following methods:

==== Configuration file ====

Download the Trillium host file, right-click on host_trillium.xml and select Save as.
Depending on the OS you are using on your local machine,

* on Linux/Mac, place the file in >~/.visit/hosts/;
* on Windows, place the file in My Documents\VisIt 2.13.0\hosts\.

Restart VisIt and check that the Trillium profile is available under Options -> Host Profiles.

==== Manual configuration ====

Open VisIt in your computer, go to the Options menu, and click on Host Profiles.
Then click on New Host and select:

 Host nickname = trillium
 Remote host name = trillium.alliancecan.ca
 Username = Enter_Your_OWN_username_HERE
 Path to VisIt installation = /scinet/trillium/software/2018a/opt/base/visit/2.13.1

Check Tunnel data connections through SSH and click on Apply.

At the top of the window, click on the Launch Profiles tab and click on New Profile. Select the appropriate profile:

* login to connect through a login node and access data;
* slurm to use compute nodes as rendering engines.

Then click on the Parallel tab and set the Launch parallel engine. For the Slurm profile, you will need to set the parameters as seen below:

Finally, after you are done with these changes, go to the Options menu and select Save settings, so that your changes are saved and available the next time you launch VisIt.

== Client-server visualization on a cloud==

=== Prerequisites ===

The Cloud Quick Start Guide explains how to launch a new virtual machine (VM). Once you log into the VM, you will need to install some additional packages to be able to compile ParaView or VisIt. For example, on a CentOS VM you can type

  sudo yum install xauth wget gcc gcc-c++ ncurses-devel python-devel libxcb-devel
  sudo yum install patch imake libxml2-python mesa-libGL mesa-libGL-devel
  sudo yum install mesa-libGLU mesa-libGLU-devel bzip2 bzip2-libs libXt-devel zlib-devel flex byacc
  sudo ln -s /usr/include/GL/glx.h /usr/local/include/GL/glx.h

If you have your own private-public SSH key pair (as opposed to the cloud key), you may want to copy the public key to the VM to simplify logins, by issuing the following command on your laptop

  cat ~/.ssh/id_rsa.pub | ssh -i ~/.ssh/cloudwestkey.pem centos@vm.ip.address 'cat >>.ssh/authorized_keys'

=== Compiling VisIt with OSMesa ===

VisIt with offscreen rendering support can be built with a single script.

 wget http://portal.nersc.gov/project/visit/releases/2.12.1/build_visit2_12_1
 chmod u+x build_visit2_12_1
 ./build_visit2_12_1 --prefix /home/centos/visit --mesa --system-python \
    --hdf4 --hdf5 --netcdf --silo --szip --xdmf --zlib

This may take a couple of hours. Once finished, you can test the installation with

 ~/visit/bin/visit -cli -nowin

This should start a VisIt Python shell.

=== Running VisIt in client-server mode ===

Start VisIt on your laptop. In Options -> Host profiles, edit the connection nickname, the VM host name, the path to the VisIt installation (/home/centos/visit) and your username on the VM; also enable tunneling through SSH. Don't forget to save your settings with Options -> Save Settings. When you open a file (File -> Open file -> Host = Arbutus) you should see the VM's filesystem. Load a file and try to visualize it. Data processing and rendering should be done on the VM, while the result and the GUI controls will be displayed on your laptop.