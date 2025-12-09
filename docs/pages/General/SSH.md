---
title: "SSH/en"
url: "https://docs.alliancecan.ca/wiki/SSH/en"
category: "General"
last_modified: "2025-09-10T18:17:15Z"
page_id: 506
display_title: "SSH"
---

`<languages />`{=html} Secure Shell (SSH) is a widely used standard to connect to remote machines securely. The SSH connection is encrypted, including the username and password. SSH is the standard way for you to connect in order to execute commands, submit jobs, check the progress of jobs, and in some cases, transfer files.

Various implementations of the SSH standard exist for most major operating systems.

- On macOS and Linux, the most widely used client is OpenSSH, a command line application installed by default.
- For recent versions of Windows, SSH is available in the PowerShell terminal, in the `cmd` prompt, or through Windows Subsystem for Linux (WSL). There are also 3rd-party SSH clients that are popular, such as [PuTTY](https://docs.alliancecan.ca/Connecting_with_PuTTY "PuTTY"){.wikilink}, [MobaXTerm](https://docs.alliancecan.ca/Connecting_with_MobaXTerm "MobaXTerm"){.wikilink}, [WinSCP](https://winscp.net/eng/download.php), and [Bitvise](https://www.bitvise.com/ssh-client-download).

To use any of these implementations of SSH successfully, you must:

- `<b>`{=html}know the name of the machine to which you want to connect.`</b>`{=html} This will be something like `fir.alliancecan.ca` or `trillium.alliancecan.ca`.
- `<b>`{=html}know your username`</b>`{=html}, typically something like `ansmith`. The `username` is `<b>`{=html}not`</b>`{=html} your CCI, like `abc-123`, nor a CCRI like `abc-123-01`, nor your email address.
- `<b>`{=html}know your password, or have an [SSH key](https://docs.alliancecan.ca/SSH_Keys "SSH key"){.wikilink}.`</b>`{=html} Your password is the same one you use to log in to [CCDB](https://ccdb.alliancecan.ca/). You may register and use an SSH key instead of a password; we highly recommend this since it provides better security.
- `<b>`{=html}be registered for [multifactor authentication](https://docs.alliancecan.ca/multifactor_authentication "multifactor authentication"){.wikilink} and have your 2nd factor available.`</b>`{=html}

From a command-line client (`<i>`{=html}e.g.`</i>`{=html} /Applications/Utilities/Terminal.app for macOS, cmd or PowerShell for Windows), use the `ssh` command like this:

For graphical clients such as MobaXterm or PuTTY, see:

- [Connecting with MobaXTerm](https://docs.alliancecan.ca/Connecting_with_MobaXTerm "Connecting with MobaXTerm"){.wikilink}
- [Connecting with PuTTY](https://docs.alliancecan.ca/Connecting_with_PuTTY "Connecting with PuTTY"){.wikilink}

The first time that you connect to a machine you\'ll be asked to store a copy of its `<i>`{=html}host key`</i>`{=html}, a unique identifier that allows the SSH client to verify, when connecting next time, that this is the same machine.

For more on generating key pairs, see:

- [SSH Keys](https://docs.alliancecan.ca/SSH_Keys "SSH Keys"){.wikilink}
  - [Generating SSH keys in Windows](https://docs.alliancecan.ca/Generating_SSH_keys_in_Windows "Generating SSH keys in Windows"){.wikilink}
  - [Using SSH keys in Linux](https://docs.alliancecan.ca/Using_SSH_keys_in_Linux "Using SSH keys in Linux"){.wikilink}

For how to use SSH to allow communication between compute nodes and the internet, see:

- [SSH tunnelling](https://docs.alliancecan.ca/SSH_tunnelling "SSH tunnelling"){.wikilink}

For how to use an SSH configuration file to simplify the login procedure, see:

- [SSH configuration file](https://docs.alliancecan.ca/SSH_configuration_file "SSH configuration file"){.wikilink}

# X11 for graphical applications {#x11_for_graphical_applications}

SSH supports graphical applications via the [X protocol](https://en.wikipedia.org/wiki/X_Window_System), now usually called \"X11\". In order to use X11 you must have an X11 server installed on your computer. Under Linux, an X11 server will normally already be installed, but users of macOS will typically need to install an external package such as [XQuartz](https://www.xquartz.org). Under Windows, MobaXterm comes with an X11 server, while for PuTTY users, there is [VcXsrv](https://sourceforge.net/projects/vcxsrv/).

Using the SSH command line, add the `-Y` option to enable X11 communications:

# Connection errors {#connection_errors}

While connecting to one of our clusters, you might get an error message such as:

- no matching cipher found
- no matching MAC found
- unable to negotiate a key exchange method
- couldn\'t agree a key exchange algorithm
- remote host identification has changed.

The last of these error messages can point to a man-in-the-middle attack, or to an upgrade of security of the cluster you are trying to connect to. If you get this, verify that the host key fingerprint mentioned in the message matches one of the host key fingerprints published at [SSH host keys](https://docs.alliancecan.ca/SSH_host_keys "SSH host keys"){.wikilink}. If it does, it is safe to continue connecting. If the host key fingerprint does not appear on our published list, terminate the connection and [contact support](https://docs.alliancecan.ca/Technical_support "contact support"){.wikilink}.

One such upgrade occurred on the Niagara cluster on May 31, 2019. See [this page](https://docs.scinet.utoronto.ca/index.php/SSH_Changes_in_May_2019) for the one-time action required from users after the security upgrade. Further upgrades of this type were made on all clusters in September/October 2019; see [SSH security improvements](https://docs.alliancecan.ca/SSH_security_improvements "SSH security improvements"){.wikilink} for more information.

If you see any of the other error messages, you will have to upgrade your OS and/or SSH client that supports strong ciphers, key exchange protocols and MAC (message authentication code) algorithms.

Here are known versions that will fail and will have to be upgraded:

- OpenSSH on CentOS/RHEL 5
- [ PuTTY](https://docs.alliancecan.ca/Connecting_with_PuTTY " PuTTY"){.wikilink} v0.64 and earlier on Windows
