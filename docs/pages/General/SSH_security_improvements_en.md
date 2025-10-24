---
title: "SSH security improvements/en"
url: "https://docs.alliancecan.ca/wiki/SSH_security_improvements/en"
category: "General"
last_modified: "2025-06-04T17:26:02Z"
page_id: 11671
display_title: "SSH security improvements"
---

`<languages />`{=html}

![SSH security improvements flowchart (click to enlarge)](https://docs.alliancecan.ca/FINAL_Flowchart_SSHD_changes_-_Summer_2019_-_v1.4_-_Copy.jpg "SSH security improvements flowchart (click to enlarge)")

[SSH](https://docs.alliancecan.ca/SSH "SSH"){.wikilink} is the software protocol that you use to connect to our clusters. It protects the security of your communication by verifying the server's identity and yours against known identification data, and by encrypting the connection. Because security risks evolve over time, we will have ended support for certain SSH options which are no longer deemed secure. You will have to make some changes on your part in order to continue using our clusters. The changes are outlined in the flowchart to the right, and explained in greater detail below.

# SSH changes (September-October 2019) {#ssh_changes_september_october_2019}

Email explaining these changes (with \"IMPORTANT\" in the subject line) was sent to all users on July 29, and with more detail on September 16.

## What is changing? {#what_is_changing}

The following SSH security improvements have been implemented on September 24, 2019 on Graham, and one week later on Béluga and Cedar:

1.  Disable certain encryption algorithms.
2.  Disable certain public key types.
3.  Regenerate the cluster\'s host keys.

If you do not understand the significance of \"encryption algorithms\", \"public keys\", or \"host keys\", do not be alarmed. Simply follow the steps outlined below. If testing indicates you need to update or change your SSH client, you may find [ page](https://docs.alliancecan.ca/SSH " page"){.wikilink} useful.

Because users do not connect to Arbutus via SSH but through a web interface, the upcoming changes do not concern them.

There were earlier, less comprehensive updates made to both Niagara (on May 31, 2019; see [here](https://docs.scinet.utoronto.ca/index.php/SSH_Changes_in_May_2019)) and Graham (early August) which would have triggered some of the same messages and errors.

## Updating your client\'s known host list {#updating_your_clients_known_host_list}

The first time you login to one of our clusters after the changes, you will probably see a warning message like the following:

    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    IT IS POSSIBLE THAT SOMEONE IS DOING SOMETHING NASTY!
    Someone could be eavesdropping on you right now (man-in-the-middle attack)!
    It is also possible that a host key has just been changed.
    The fingerprint for the ED25519 key sent by the remote host is
    SHA256:mf1jJ3ndpXhpo0k38xVxjH8Kjtq3o1+ZtTVbeM0xeCk.
    Please contact your system administrator.
    Add correct host key in /home/username/.ssh/known_hosts to get rid of this message.
    Offending ECDSA key in /home/username/.ssh/known_hosts:109
    ED25519 host key for graham.computecanada.ca has changed and you have requested strict checking.
    Host key verification failed.
    Killed by signal 1.

This warning is displayed because the host keys on the cluster (in this case [Graham](https://docs.alliancecan.ca/Graham "Graham"){.wikilink}) were changed, and your SSH client software remembers the old host keys. (It does this to prevent [\"man-in-the-middle\" attacks](https://en.wikipedia.org/wiki/Man-in-the-middle_attack).) This will happen for your SSH client on each device from which you connect, so you may see it multiple times.

You may also get a warning regarding \"DNS spoofing\", which is related to the same change.

### MobaXterm, PuTTY, or WinSCP {#mobaxterm_putty_or_winscp}

If you are using [MobaXterm](https://docs.alliancecan.ca/Connecting_with_MobaXTerm "MobaXterm"){.wikilink}, [PuTTY](https://docs.alliancecan.ca/Connecting_with_PuTTY "PuTTY"){.wikilink}, or [WinSCP](https://winscp.net/eng/download.php) as your ssh (or scp) client under Windows, the warning will appear in a pop-up window and will allow you to accept the new host key by clicking \"Yes\". **Only click yes if the fingerprint matches one listed in [SSH host key fingerprints](https://docs.alliancecan.ca/SSH_changes#SSH_host_key_fingerprints "SSH host key fingerprints"){.wikilink}** at the bottom of this page. If the fingerprint does not match any on the list, do not accept the connection, and contact [Technical support](https://docs.alliancecan.ca/Technical_support "Technical support"){.wikilink} with the details.

### macOS, Linux, GitBash or Cygwin {#macos_linux_gitbash_or_cygwin}

If you are using the command line `ssh` command on macOS, Linux, GitBash or Cygwin, you should tell your system to \"forget\" the old host key by running one of the following commands:

- Graham

`for h in 2620:123:7002:4::{2..5} 199.241.166.{2..5} {gra-login{1..3},graham,gra-dtn,gra-dtn1,gra-platform,gra-platform1}.{sharcnet,computecanada}.ca; do ssh-keygen -R $h; done`

- Cedar

`for h in 206.12.124.{2,6} cedar{1,5}.cedar.computecanada.ca cedar.computecanada.ca; do ssh-keygen -R $h; done`

- Beluga

`for h in beluga{,{1..4}}.{computecanada,calculquebec}.ca 132.219.136.{1..4}; do ssh-keygen -R $h; done`

- Mp2

`for h in ip{15..20}-mp2.{computecanada,calculquebec}.ca 204.19.23.2{15..20}; do ssh-keygen -R $h; done`

Afterwards, the next time you ssh to the cluster you\'ll be asked to confirm the new host keys, e.g.:

`$ ssh graham.computecanada.ca`\
`The authenticity of host 'graham.computecanada.ca (142.150.188.70)' can't be established.`\
`ED25519 key fingerprint is SHA256:mf1jJ3ndpXhpo0k38xVxjH8Kjtq3o1+ZtTVbeM0xeCk.`\
`ED25519 key fingerprint is MD5:bc:93:0c:64:f7:e7:cf:d9:db:81:40:be:4d:cd:12:5c.`\
`Are you sure you want to continue connecting (yes/no)? `

**Only type yes if the fingerprint matches one listed in the [SSH host key fingerprints](https://docs.alliancecan.ca/SSH_changes#SSH_host_key_fingerprints "SSH host key fingerprints"){.wikilink}** at the bottom of this page. If the fingerprint does not match any on the list below, do not accept the connection, and contact [Technical support](https://docs.alliancecan.ca/Technical_support "Technical support"){.wikilink} with the details.

## Troubleshooting

See the list of [SSH host key fingerprints](https://docs.alliancecan.ca/SSH_changes#SSH_host_key_fingerprints "SSH host key fingerprints"){.wikilink}.

### My SSH key no longer works {#my_ssh_key_no_longer_works}

If you\'re being asked for a password, but were using SSH keys previously on the same system, it\'s likely because 1024-bit DSA & RSA keys have been disabled.

You need to generate a new stronger key. The process for doing this depends on the operating system you use, either [Windows](https://docs.alliancecan.ca/Generating_SSH_keys_in_Windows "Windows"){.wikilink} or [Linux/macOS](https://docs.alliancecan.ca/Using_SSH_keys_in_Linux "Linux/macOS"){.wikilink}. Those instructions also describe how to add your client\'s new public key to the remote host, so that you can authenticate with the key rather than needing to provide a password.

### I can\'t connect! {#i_cant_connect}

If you see any of the following error messages:

`Unable to negotiate with 142.150.188.70 port 22: no matching cipher found.`\
`Unable to negotiate with 142.150.188.70 port 22: no matching key exchange method found.`\
`Unable to negotiate with 142.150.188.70 port 22: no matching mac found.`

you need to upgrade your SSH client to one of the compatible clients shown below.

### Which clients are compatible with the new configuration? {#which_clients_are_compatible_with_the_new_configuration}

**The list below is not exhaustive**, but we have tested the configuration with the following clients. Earlier versions of these clients may or may not work. We recommend that you update your operating system and SSH client to the latest version compatible with your hardware.

#### Linux clients {#linux_clients}

- OpenSSH_7.4p1, OpenSSL 1.0.2k-fips (CentOS 7.5, 7.6)
- OpenSSH_6.6.1p1 Ubuntu-2ubuntu2.13, OpenSSL 1.0.1f (Ubuntu 14)

#### OS X clients {#os_x_clients}

You can determine the exact version of your SSH client on OS X using the command `ssh -V`.

- OpenSSH 7.4p1, OpenSSL 1.0.2k (Homebrew)
- OpenSSH 7.9p1, LibreSSL 2.7.3 (OS X 10.14.5)

#### Windows clients {#windows_clients}

- [MobaXterm Home Edition](https://docs.alliancecan.ca/Connecting_with_MobaXTerm "MobaXterm Home Edition"){.wikilink} v11.1
- [PuTTY](https://docs.alliancecan.ca/Connecting_with_PuTTY "PuTTY"){.wikilink} 0.72
- Windows Services for Linux (WSL) v1
  - Ubuntu 18.04 (OpenSSH_7.6p1 Ubuntu-4ubuntu0.3, OpenSSL 1.0.2n)
  - openSUSE Leap 15.1 (OpenSSH_7.9p1, OpenSSL 1.1.0i-fips)

#### iOS clients {#ios_clients}

- Termius, 4.3.12

# SSH host key fingerprints {#ssh_host_key_fingerprints}

To retrieve the host fingerprints remotely, one can use the following commands:

`ssh-keyscan ``<hostname>`{=html}` | ssh-keygen -E md5 -l -f -`\
`ssh-keyscan ``<hostname>`{=html}` | ssh-keygen -E sha256 -l -f -`

Listed below are the SSH fingerprints for our clusters. If the fingerprint you get does not match any on the list below, do not accept the connection, and contact [Technical support](https://docs.alliancecan.ca/Technical_support "Technical support"){.wikilink} with the details.
