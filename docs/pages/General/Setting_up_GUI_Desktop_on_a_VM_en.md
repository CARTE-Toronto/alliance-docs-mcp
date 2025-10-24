---
title: "Setting up GUI Desktop on a VM/en"
url: "https://docs.alliancecan.ca/wiki/Setting_up_GUI_Desktop_on_a_VM/en"
category: "General"
last_modified: "2023-02-27T21:02:02Z"
page_id: 15396
display_title: "Setting up GUI Desktop on a VM"
---

`<languages />`{=html}

Some software that you can install on your virtual machine (VM, or instance) are only, or best accessed, through their graphical user interface (GUI). It is possible to use a GUI through SSH + X11 forwarding. However, you may observe better performance when using VNC to connect to a remote desktop running on your VM.

Below, we outline steps for setting a remote desktop with VNC. Please note that these instructions are for a VM running a Ubuntu operating system.

1.  Install a GUI Desktop on your VM.\
    There are lots of different Desktop packages available. For example some common Desktop packages available for the Ubuntu operating system are:
    - \[<https://ubuntuunity.org/>\| ubuntu-unity-desktop\]
    - \[<https://ubuntu-mate.org/>\| ubuntu-mate-desktop\]
    - \[<https://lubuntu.net/>\| lubuntu-desktop\]
    - \[<https://xubuntu.org/screenshots/>\| xubuntu-desktop\]
    - \[<https://www.xfce.org/>\| xfce4\]
    - ubuntu-desktop
    - \[<https://kde.org/plasma-desktop/>\| kde-plasma-desktop\]
    - ubuntu-desktop-minimal
    - \[<https://en.wikipedia.org/wiki/Cinnamon_(desktop_environment)>\| cinnamon\]
    - \[<https://ice-wm.org/>\| icewm\]

    [This article](https://cloudinfrastructureservices.co.uk/best-ubuntu-desktop-environments) shows a few of these different desktops. Below are the commands to install the MATE desktop.

    During the installation of the `ubuntu-mate-desktop` package it will ask you to choose the default display manager, a good option is \[<https://en.wikipedia.org/wiki/LightDM>\|`lightdm`\]. Installing the `ubuntu-mate-desktop` package can take a fair amount of time (something like 15-30 mins).
2.  Install TigerVNC server.\
    This software runs on your VM and allows you to use the GUI desktop you installed in step 1. remotely using a client software.
    This command will install the TigerVNC server and some supporting software. For details about using VNC servers and clients see our docs on [VNC](https://docs.alliancecan.ca/VNC "VNC"){.wikilink}.
3.  Start the vnc server
    `The first time you start a vnc server it will ask you to set a password. This password is used later when connecting to the vnc desktop. You don't need a view-only password. The ``vncpasswd`` command can later be used to change your password.`
4.  Test your connection by opening port `5901` (see [ security groups](https://docs.alliancecan.ca/Managing_your_cloud_resources_with_OpenStack#Security_Groups " security groups"){.wikilink} for more information about opening ports to your VMs with OpenStack) and connecting using a VNC viewer, for example [TigerVNC](https://tigervnc.org/). However, this is not a secure connection; data sent to and from your VM will not be encrypted. This is only meant to test your server-client connection before connecting securely with an SSH tunnel (the next step). If you are confident in your ability to setup an SSH tunnel, you may skip this step.
5.  Connect using an SSH tunnel (see [SSH_tunnelling](https://docs.alliancecan.ca/SSH_tunnelling "SSH_tunnelling"){.wikilink}). There is [an example of creating an SSH tunnel to a VNC server running on a compute node of one of our clusters](https://docs.alliancecan.ca/VNC#Compute_Nodes "an example of creating an SSH tunnel to a VNC server running on a compute node of one of our clusters"){.wikilink}. Below are instructions for connecting using an SSH tunnel for linux or mac:
    - Open your terminal
    - Type the following in your local terminal: `SSH -i filepathtoyoursshkey/sshprivatekeyfile.key -L5901:localhost:5901 ubuntu@ipaddressofyourVM`
    - Start your VNC viewer.
    - In the VNC server field enter: `localhost:5901`.
    - Your GUI desktop for your remote session should now open
6.  Close port `5901`. Once you are connected to your VNC server using an SSH tunnel, you no longer require port 5901 open so it is recommended that you remove this rule from your security groups. (see [security groups](https://docs.alliancecan.ca/Managing_your_cloud_resources_with_OpenStack#Security_Groups "security groups"){.wikilink} for more information).
7.  Once you are finished using the remote desktop you may stop the vncserver with:
