---
title: "Configuring WSL as a ControlMaster relay server/en"
url: "https://docs.alliancecan.ca/wiki/Configuring_WSL_as_a_ControlMaster_relay_server/en"
category: "General"
last_modified: "2024-04-03T17:28:39Z"
page_id: 25216
display_title: "Configuring WSL as a ControlMaster relay server"
---

`<languages />`{=html}

With this procedure you can leverage ControlMaster under WSL so you may log into the clusters with several apps under native Windows for a certain period without having to use multifactor authentication for every session.

### Install Linux on Windows with WSL {#install_linux_on_windows_with_wsl}

Please follow this link for more detailed instructions:

[`https://docs.alliancecan.ca/wiki/Windows_Subsystem_for_Linux_(WSL)`](https://docs.alliancecan.ca/wiki/Windows_Subsystem_for_Linux_(WSL))

This setup assumes the following on the sample config files:

- you selected Ubuntu as your distribution
- the hostname for the WSL instance is `<i>`{=html}ubuntu`</i>`{=html}: `<i>`{=html}/etc/hostname`</i>`{=html} contains `<i>`{=html}ubuntu`</i>`{=html} and `<i>`{=html}/etc/hosts`</i>`{=html} contains `<i>`{=html}127.0.0.1 localhost ubuntu`</i>`{=html}
- the Windows system is named `<i>`{=html}smart`</i>`{=html} and the login name is `<i>`{=html}jaime`</i>`{=html}
- the user name on the Ubuntu VM is also `<i>`{=html}jaime`</i>`{=html}
- the Alliance user name is `<i>`{=html}pinto`</i>`{=html} and we want to connect to Cedar

### Install additional packages {#install_additional_packages}

     sudo apt update && sudo apt upgrade -y
     sudo apt install openssh-server -y

You may log in from Windows to Ubuntu with `ssh localhost`.

### General idea of the setup {#general_idea_of_the_setup}

    [ssh client] ----> [ssh relay server] ----> [ssh target server]
    your Windows     modified authorized_keys     using cedar for
      machine          in your Ubuntu VM           this exercise
     <i>smart</i>        <i>ubuntu</i>                 Cedar

### Log into the Ubuntu VM and create a `<i>`{=html}custom_ssh`</i>`{=html} folder {#log_into_the_ubuntu_vm_and_create_a_custom_ssh_folder}

    jaime@ubuntu:~$ cat custom_ssh/sshd_config
    Port 2222
    HostKey /home/jaime/custom_ssh/ssh_host_ed25519_key
    HostKey /home/jaime/custom_ssh/ssh_host_rsa_key
    AuthorizedKeysFile /home/jaime/custom_ssh/authorized_keys
    ChallengeResponseAuthentication no
    UsePAM no
    Subsystem sftp /usr/lib/openssh/sftp-server
    PidFile /home/jaime/custom_ssh/sshd.pid

You may copy the *ssh_host* keys from */etc/ssh* with:

    sudo cp /etc/ssh/ssh_host_ed25519_key /home/jaime/custom_ssh/

### Customize `<i>`{=html}.ssh/config`</i>`{=html} on Ubuntu {#customize_.sshconfig_on_ubuntu}

    jaime@ubuntu:~$ cat ~/.ssh/config
    Host cedar
        ControlPath ~/.ssh/cm-%r@%h:%p
        ControlMaster auto
        ControlPersist 10m
        HostName cedar.alliancecan.ca
        User pinto

### Customize the authorized keys {#customize_the_authorized_keys}

    jaime@ubuntu:~/custom_ssh$ cat /home/jaime/custom_ssh/authorized_keys
    ssh-ed25519 AAAZDINzaC1lZDI1NTE5AAC1lZDIvqzlffkzcjRAaMQoTBrPe5FxlSAjRAaMQyVzN+A+

Use the same public SSH key that you uploaded to CCDB.

### Now start the sshd server on Ubuntu {#now_start_the_sshd_server_on_ubuntu}

    jaime@ubuntu:~/custom_ssh$ /usr/sbin/sshd -f ${HOME}/custom_ssh/sshd_config

Make sure you start the server as yourself, not as root. You will also need to start the sshd server every time you restart your computer, or after closing or restarting WSL.

### Customize `<i>`{=html}.ssh/config`</i>`{=html} on `<i>`{=html}smart`</i>`{=html} with `RemoteCommand` {#customize_.sshconfig_on_smart_with_remotecommand}

    jaime@smart ~/.ssh cat config
    Host ubuntu
            Hostname localhost
            RemoteCommand ssh cedar

### You are now ready to try to log into Cedar {#you_are_now_ready_to_try_to_log_into_cedar}

    jaime@smart ~
    $ ssh -t ubuntu -p 2222
    Enter passphrase for key '/home/jaime/.ssh/id_ed25519':
    Last login: Fri Mar 22 10:50:12 2024 from 99.239.174.157
    ================================================================================
    Welcome to Cedar! / Bienvenue sur Cedar!
    ...
    ...
    ...
    [pinto@cedar1 ~]$

### Alternative setup {#alternative_setup}

There is another way in which you could customize the authorized keys on Ubuntu and the `<i>`{=html}\~/.ssh/config`</i>`{=html} on Windows such that it may work better for some Windows GUI apps that don\'t let you explicitly set the `RemoteCommand` (such as WinSCP). In this case you set the `RemoteCommand` on the public key:

    jaime@ubuntu:~/custom_ssh$ cat /home/jaime/custom_ssh/authorized_keys
    command="ssh cedar" ssh-ed25519 AAAZDINzaC1lZDI1NTE5AAC1lZDIvqzlffkzcjRAaMQoTBrPe5FxlSAjRAaMQyVzN+A+

    jaime@smart ~/.ssh cat config
    Host ubuntu
            Hostname localhost
            #RemoteCommand ssh cedar

You may still use `ssh ubuntu -p 2222` after that from a shell on Windows.

### Setup with MobaXterm {#setup_with_mobaxterm}

![](MobaXterm-setup.jpg "MobaXterm-setup.jpg")

![](MobaXterm-VSL-localdriveC.jpg "MobaXterm-VSL-localdriveC.jpg")
