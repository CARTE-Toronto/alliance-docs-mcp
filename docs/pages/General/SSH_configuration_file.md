---
title: "SSH configuration file/en"
url: "https://docs.alliancecan.ca/wiki/SSH_configuration_file/en"
category: "General"
last_modified: "2025-02-03T17:28:41Z"
page_id: 23199
display_title: "SSH configuration file"
---

Parent page: SSH

On Linux and macOS, you can modify your local SSH configuration file to change the default behaviour of ssh and simplify the login procedure. For example, if you want to log into narval.alliancecan.ca as username using an SSH key, you may need to use the following command:

To avoid having to type this command each time you want to connect to Narval, add the following to ~/.ssh/config on your local machine:
  Host narval
    User username
    HostName narval.alliancecan.ca
    IdentityFile ~/.ssh/your_private_key

You can now log into Narval by typing

This also changes the behaviour of sftp, scp, and rsync and you can now  transfer files by typing for example

If you frequently log into different clusters, modify the above Host block as follows instead of adding individual entries for each cluster separately:
  Host narval beluga graham cedar
    [...]
    HostName %h.alliancecan.ca
    [...]
Note that you need to install your public  SSH key on each cluster separately or use CCDB.

Note that other options of the ssh commands have corresponding parameters that you can put in your ~/.ssh/config file on your machine. In particular, the command line options
* -X (X11 forwarding)
* -Y (trusted X11 forwarding)
* -A (agent forwarding)
can be set through your configuration file by adding lines with
* ForwardX11 yes
* ForwardX11Trusted yes
* ForwardAgent yes
in the corresponding sections of your configuration file. However, we do not recommend doing so in general, for these reasons:
* Enabling X11 forwarding by default for all of your connections can slow down your sessions, especially if your X11 client on your computer is misconfigured.
* Enabling trusted X11 forwarding comes with a risk. Should the server to which you are connecting to be compromised, a privileged user (root) could intercept keyboard activity on your local computer. Use trusted X11 forwarding only when you need it.
* Similarly, while forwarding your SSH agent is convenient and more secure than typing a password on a remote computer, it also comes with a risk. Should the server to which you are connecting to be compromised, a privileged user (root) could use your agent and connect to another host without your knowledge. Use agent forwarding only when you need it. We also recommend that, if you use this feature, you should combine it with ssh-askpass so that any use of your SSH agent triggers a prompt on your computer, preventing usage of your agent without your knowledge.