---
title: "SSH Keys/en"
url: "https://docs.alliancecan.ca/wiki/SSH_Keys/en"
category: "General"
last_modified: "2025-08-29T12:53:58Z"
page_id: 770
display_title: "SSH Keys"
---

`<languages />`{=html}

`<i>`{=html}Parent page: [SSH](https://docs.alliancecan.ca/SSH "SSH"){.wikilink}`</i>`{=html}

## What are SSH keys? {#what_are_ssh_keys}

SSH relies on [public key cryptography](https://en.wikipedia.org/wiki/Public-key_cryptography) (PK) for its security.

PK is based on a `<i>`{=html}key pair,`</i>`{=html} which consists of a private part, to be kept secret, and a public part, which can be distributed freely. Anyone can use the public key to encode a message, but the message can only be decoded with the private part. This is why PK is sometimes described as `<i>`{=html}asymmetric encryption.`</i>`{=html}

PK can also be used to verify identities: if someone is claiming to be Alice, then a second party, Bob, can send Alice a message encoded with Alice\'s public key. If the person claiming to be Alice can tell Bob what is in the message, then that person has access to Alice\'s private key. In this sense, possession of a private key establishes identity.

PK systems are the basis for the SSL and TLS protocols that protect most internet traffic, such as HTTPS websites.

On our systems, PK is used in SSH several ways:

- When connecting to our systems, your SSH client normally uses our system\'s public key to ensure that it has connected to the real (authentic) server.
- PK is used to establish an encrypted session so that all following traffic is secure from eavesdropping.
- The remote server can use your public key to verify your identity, that is, to allow you to log in.

`<b>`{=html}We strongly recommend using PK to connect to our systems.`</b>`{=html} Using PK is almost always more secure than using a password.

## Generating an SSH Key {#generating_an_ssh_key}

Using PK requires some initial configuration, but once this is done it is both more secure and more convenient. To use PK, you must generate a key pair and install the public key on the systems you intend to connect with.

You should generate a key pair on your own system, or on a system you believe is secure. The private part of the key pair should never be shared or copied on another computer, as is the case with any password.

When generating a key pair, you will be prompted for a `<i>`{=html}passphrase`</i>`{=html}. This is a string that is used to encrypt the private key. You should `<b>`{=html}provide a strong passphrase`</b>`{=html} that is memorable. We recommend 15 characters or more. This passphrase offers protection if the private key file is stolen.

The specific process to generate an SSH key pair depends on the operating system you use. For the Windows PuTTY or MobaXterm clients, see [Generating SSH keys in Windows](https://docs.alliancecan.ca/Generating_SSH_keys_in_Windows "Generating SSH keys in Windows"){.wikilink}. For a Unix-like environment (Linux, Mac, Windows Subsystem for Linux or Cygwin), see [Using SSH keys in Linux](https://docs.alliancecan.ca/Using_SSH_keys_in_Linux "Using SSH keys in Linux"){.wikilink}. In addition if you are using the cloud, OpenStack provides a method for creating key pairs: See the [SSH key pair](https://docs.alliancecan.ca/Cloud_Quick_Start#SSH_key_pair "SSH key pair"){.wikilink} section on the Cloud Quick Start page.

## Installing your key {#installing_your_key}

### Using CCDB {#using_ccdb}

To install the key, you must make the target/destination system aware of the public part of your key. In March 2021, we added a convenient way to do this.

You should follow the steps:

`<b>`{=html}STEP 1`</b>`{=html} - Go to CCDB as indicated below

[`https://ccdb.alliancecan.ca/ssh_authorized_keys`](https://ccdb.alliancecan.ca/ssh_authorized_keys)

\
Or via the CCDB menu:\
![](Ccdb-keys-fig1.png "Ccdb-keys-fig1.png"){width="200"}

`<b>`{=html}STEP 2`</b>`{=html} - Grab your SSH public key

Since the public key is plain text, you can examine it under Windows by opening the file with Notepad. Under Linux/MacOS enter `cat .ssh/id_rsa.pub` on the command line.

In both cases, you will see something like this, as a very long single line:

`ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQC3qeojDkUShnPTq9pI3cCZe+jgD6RKA/6CPsJIWZ8MqbX6wkk6`\
`hHgKqKd2/9d7cj8e03Cfv4JLoD++P9fUPE3UyYrP/uVi4zytp5rmIZI4Qo1LvD1Obs0e78y0Dp7pfG1EHTYdn0p8`\
`zHa0lNLOrZMmzDP0UMVdf4WKuv3Th2K/35yQ7DVIei6X/33Dcmsqd8bTXq7aFlw2y4sa/CHs314G6WYuJ9cTXtsW`\
`Dlc9oWJuOVqILJLeGQpl3BVEM9aKcYksqLMV1UlZF8bHbry0PKCnrJrNMzVulWfnmSOZ+lPcV32OsDRkFaKsdxPy`\
`+PxywieC86mxy1v216jeOdHnhLfOJc/VYDqf4egxReSCb3WOucHBB5J1jtDt47GuamKs+F2T7obqCb0J6oTyzgVF`\
`RIZryxwvh5fQF5jk3LsBGsbhe9GYwDAk54GV6I0rWnXp56mJZjO4JCRQGbwLAJVxH4a7UrBmba2pRcZxEZmbIcBB`\
`Sjb9KPECtaxiY/aty39077DCmcLCmzeOgBdh0zIkdBYu+OJ65MFKMxzoJWPDbZIcbSRGHEVhDnBZMNj1OiJS+E2D`\
`A+F0tPH7J+xox1vUoXGAI0cNv+s/nlVRuOZoZjhk6s7tLXeVcToc+Y9Wqx8fdL7D4FegWwB9lsVhpfC4NaA9R8Ao`\
`OfJUwHSNqUc6SfIt7w== user@machine`

Note that you must generate your own pair of public and private keys; you cannot copy the above example for your own use. If the format observed is different (which is the case when the key is generated with PuTTY for instance), you can open PuTTY Key Generator or MobaxTerm SSH Key Generator, go to `<i>`{=html}File \--\> Load private key`</i>`{=html} and grab the content under `<i>`{=html}Public key for pasting into OpenSSH authorized_keys`</i>`{=html} as shown in [this image](https://docs.alliancecan.ca/wiki/File:Puttygen2.png).

For instance, this is a public key in PEM format that would need to be converted before going to STEP 3:

`-----BEGIN RSA PUBLIC KEY-----`\
`MIIBCgKCAQEAxFm+Fbs+szeV2Vg2T5ufg8az0jD9DD/A0iNLKef2/0gPULn1ebFQ`\
`SvQwts5ZGcza9t6l7fSKObz8FiAwXn+mdmXrxx3fQIepWa2FeCNbTkiKTTpNmERw`\
`H0v3RR3DpJd8cpg5jdJbINlqDUPdqXxZDPIyZuHbEYUiSrb1v5zscVdgVqhJYi9O`\
`OiEj7dPOLp1ko6s7TSgY8ejGnbmUL/gl+/dfhMNKdhLXMXWByucF1577rfAz3qPn`\
`4JMWrG5TCH7Jj8NpIxFhkV9Qjy40Ml81yDqMlbuE9CUZzVhATe8MdIvcXUQej8yl`\
`ddmNnAXmfTDwUd5cJ/VSMaKeq6Gjd/XDmwIDAQAB`\
`-----END RSA PUBLIC KEY-----`

`<b>`{=html}STEP 3`</b>`{=html} - Upload your SSH public key into CCDB form

You can then paste this `<b>`{=html}public`</b>`{=html} portion of your key into the CCDB form. You should also provide a description for this key so you can keep track of them.\
![](Ccdb-keys-fig2.png "Ccdb-keys-fig2.png"){width="400"}

After clicking `<i>`{=html}Add Key,`</i>`{=html} your SSH key will show up in the section below.\
![](Ccdb-keys-fig3.png "Ccdb-keys-fig3.png"){width="400"}

Once your public key is loaded into CCDB this way, you can use it to log in to any of our clusters. However, our OpenStack cloud systems cannot access the keys entered in the CCDB.

Changes to public keys will often propagate to clusters in a few minutes; in some cases however, it can take up to approximately 30 minutes.

Note: Public keys in RFC4716 or PKCS8 formats will look similar to PEM, with small variations in the header and footer lines.

### Using the authorized_keys file {#using_the_authorized_keys_file}

The CCDB method described above makes your public key available on all of our HPC systems. This is convenient, and is often desired.

However, there may be circumstances in which you want to install a key only on a specific system. You can do this by adding the key to a file in your /home directory on that system. For instance, to install a key that only works on Fir, copy your public key into the file `~/.ssh/authorized_keys` on Fir. This will allow you to log in to any of Fir\'s login nodes using PK. On our systems (or any other with OpenSSH) the `ssh-copy-id` command is the most convenient way to do this.

` ssh-copy-id -i alliance-key username@fir.alliancecan.ca `

The `authorized_keys` mechanism is standard, and almost universally used on the internet. It is however somewhat fragile: Specifically, SSH is quite picky about the permissions on the `authorized_keys` file, as well as your /home directory and the `.ssh` subdirectory. This is described further in [using SSH keys in Linux](https://docs.alliancecan.ca/Using_SSH_keys_in_Linux "using SSH keys in Linux"){.wikilink}.

## Using a key agent {#using_a_key_agent}

Although it\'s important to secure your private key by encrypting it with the passphrase, it is inconvenient to have to enter your passphrase every time you use the key. Rather than leaving the private key unencrypted, we strongly suggest using an SSH key agent. You type the passphrase when starting up the agent, after which the agent supplies the private key for new connections. This avoids storing the unencrypted private key on permanent storage, where it is more vulnerable to being stolen or copied.

## Options for key generation {#options_for_key_generation}

When you generate a key, the default settings are usually sufficient. However, here are a few options which may be of interest. We demonstrate these options here using `ssh-keygen` as described in [Using SSH keys in Linux](https://docs.alliancecan.ca/Using_SSH_keys_in_Linux "Using SSH keys in Linux"){.wikilink}, but the same options are available if you are using a graphical interface as described in [Generating SSH keys in Windows](https://docs.alliancecan.ca/Generating_SSH_keys_in_Windows "Generating SSH keys in Windows"){.wikilink}.

- You can specify a comment for the key, which may be helpful if you have multiple keys.

`ssh-keygen -C 'Alliance systems'`

- You can also choose the name of the key file.

`ssh-keygen -f alliance-key`

This produces a file `alliance-key` containing the private part, and `alliance-key.pub` for the public part. If you do this, though, you may have to use the `-i` option to specify the name of the key when logging in, like this: `ssh -i alliance-key user@host`

- There are sometimes reasons to choose a different key type (rather than the RSA default).

`ssh-keygen -t ed25519`

- You can strengthen certain key types, such as RSA, by setting a longer key length.

`ssh-keygen -t rsa-sha2-512 -b 4096`

## SSH key constraints {#ssh_key_constraints}

The public key syntax permits you to provide a number of very useful constraints that limit what the key is allowed to do. By default, a public key installed without constraints can do anything. For instance, this public key

` restrict,command="squeue --me" ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGhczaUoV6SzR2VEf9Rp4/P9xHVU8S72CKHrwKU+Yntx`

can only perform one simple operation (showing whether you have any jobs in Slurm). An interesting example is

` restrict,command="/usr/libexec/openssh/sftp-server" ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGhczaUoV6SzR2VEf9Rp4/P9xHVU8S72CKHrwKU+Yntx`

which allows the key to be used only for SFTP, which is how sshfs works.

The key constraint can also limit which hosts can connect using the key.

` restrict,from="d24-141-114-17.home.cgocable.net" ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGhczaUoV6SzR2VEf9Rp4/P9xHVU8S72CKHrwKU+Yntx`

Limiting by hosts is a powerful way to minimize the danger posed by a key being compromised. In this case, the `restrict` keyword turns off `pty allocation`, which makes an interactive session behave peculiarly. For a source-constrained interactive session

` restrict,from="d24-141-114-17.home.cgocable.net",pty ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGhczaUoV6SzR2VEf9Rp4/P9xHVU8S72CKHrwKU+Yntx`

allows pty allocation.

There are a large number of these key constraints, which are documented in the `<i>`{=html}sshd man`</i>`{=html} page (`<i>`{=html}man sshd`</i>`{=html} on a Linux system).

## Best practices for key pairs {#best_practices_for_key_pairs}

- Keys should be unique to a given user account. They should not be used by multiple people.
- If you have to use a shared computer, store your private key where only you are allowed access. If such secure storage area is not available, use an external storage such as a USB key.
- Always encrypt your private key with a passphrase. We recommend 15 characters or more for the passphrase.
- Do not share your private key.
- Do not copy your private key to a remote system. Your private key should not leave your workstation!
- If you have several laptops, you can create dedicated SSH key pairs for each of them.
- If you have several pairs, you may wish to name the keys. For example, `Laptop_RSA4096`.
  - If you name a key you must use the `-i` option to specify the key name when logging in, like this: `ssh -i Laptop_RSA4096 username@host`
- Use `ssh-agent` to make encrypted keys convenient.
- If you use agent forwarding, use `ssh-askpass` too.
- Apply [ constraints](https://docs.alliancecan.ca/SSH_Keys#SSH_key_constraints " constraints"){.wikilink} to your public key to limit its scope.

Here are some links to short videos on setting up SSH keys:

- [Faster and more secure SSH](https://www.youtube.com/watch?v=mRdqM1dgf3Q&feature=youtu.be)
- [Using SSH Keys on Windows](https://www.youtube.com/watch?v=q9YA5H53IHQ)
- [Using SSH Keys on Mac](https://www.youtube.com/watch?v=E8-CfvumJBo)
- [Using SSH Keys on Linux](https://www.youtube.com/watch?v=owt-tYEQOZ0)

## Revoked SSH Keys {#revoked_ssh_keys}

An SSH key may be `<b>`{=html}revoked`</b>`{=html} for several reasons, such as in cases of identified SSH key sharing or if it is believed that the private component of the SSH key may have become compromised.

This can occur through various means, such as `<i>`{=html}improper key management, theft`</i>`{=html}, or a `<i>`{=html}security breach`</i>`{=html}.

This poses a significant security risk, as a malicious actor may use the key to gain unauthorized access to systems and sensitive data. To mitigate the risk associated with compromised SSH keys, the Digital Research Alliance of Canada maintains a `<b>`{=html}Key Revocation List (KRL)`</b>`{=html}.

The Key Revocation List is a registry of SSH keys that are no longer trusted, or are otherwise considered invalid. No SSH key on the list can be used to access Alliance services.

If you discover that your SSH key has been revoked it is `<i>`{=html}imperative`</i>`{=html} that you take immediate action.

- Replace the revoked key with a new one to ensure that you can connect securely to Alliance services.
- Remove the revoked key from `<i>`{=html}all`</i>`{=html} services (Alliance or other) to prevent unauthorized access or data breaches.

If you believe that your SSH key has been listed on the Key Revocation List in error, or if you have concerns or questions related to key revocation and access to Alliance services, contact our [Technical support](https://docs.alliancecan.ca/Technical_support "Technical support"){.wikilink} for assistance. They will be able to guide you through the resolution process and help ensure the continued security of your digital interactions with our services. Security is of paramount importance, and swift action in response to a revoked SSH key is essential to maintain the integrity of digital research and collaboration.
