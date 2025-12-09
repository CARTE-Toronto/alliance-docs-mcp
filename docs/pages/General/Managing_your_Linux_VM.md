---
title: "Managing your Linux VM/en"
url: "https://docs.alliancecan.ca/wiki/Managing_your_Linux_VM/en"
category: "General"
last_modified: "2023-05-01T20:09:02Z"
page_id: 21279
display_title: "Managing your Linux VM"
---

The majority of researchers use the Linux Operating System on their VMs.  Common Linux distributions used are AlmaLunix, CentOS, Debian, Fedora, and Ubuntu.  This page will help you with some common tasks to manage your Linux VM.  VMs can also run the Microsoft Windows operating system.  Some Windows management tasks are described  here.

=Linux VM user management=
There are a number of ways to allow more than one person to log into a VM. We recommend creating new user accounts and adding public SSH Keys to these accounts.

==Creating a user account and keys==
A new user account can be created on Ubuntu with the command  To be able to connect, the new user will need to have a key pair, see generating SSH keys in Windows or creating a key pair in Linux or Mac depending on the operating system they will be connecting from. Then, their public key must be added to /home/USERNAME/.ssh/authorized_keys on the VM, ensuring permissions and ownership are correct as described in steps 2 and 3 of Connecting using a key pair.

==Granting admin privileges==
In Ubuntu, administrative or root user privileges can be given to a new user with the command

which opens an editor where a line like
 USERNAME ALL=(ALL) NOPASSWD:ALL
can be added. For more detailed information about the visudo command and how to edit this file see this digitalocean tutorial.

==Dealing with system and security issues==
See our guides for how to
*  recover data from a compromised VM
*  recover your VM from the dashboard