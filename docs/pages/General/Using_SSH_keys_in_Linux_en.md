---
title: "Using SSH keys in Linux/en"
url: "https://docs.alliancecan.ca/wiki/Using_SSH_keys_in_Linux/en"
category: "General"
last_modified: "2023-02-02T15:47:02Z"
page_id: 647
display_title: "Using SSH keys in Linux"
---

`<languages />`{=html}

*Parent page: [SSH](https://docs.alliancecan.ca/SSH "SSH"){.wikilink}*

# Creating a key pair {#creating_a_key_pair}

Before creating a new key pair, check to see if you already have one. If you do, but can\'t remember where you\'ve used it, it\'s better to create a fresh one, since you shouldn\'t install a key of unknown security.

Key pairs are typically located in the `.ssh/` directory in your /home directory. By default, a key is named with an \"id\_\" prefix, followed by the key type (\"rsa\", \"dsa\", \"ed25519\"), and the public key also has a \".pub\" suffix. So a common example is `id_rsa` and `id_rsa.pub`. A good practice is to give it a name that is meaningful to you and identify on which system the key is used.

If you do need a new key, you can generate it with the `ssh-keygen` command:

``` console
[name@yourLaptop]$  ssh-keygen -t ed25519
```

or

``` console
[name@yourLaptop]$ ssh-keygen -b 4096 -t rsa
```

(This example explicitly asks for a 4-kbit RSA key, which is a reasonable choice.)

The output will be similar to the following:

``` console
Generating public/private rsa key pair.
Enter file in which to save the key (/home/username/.ssh/id_rsa):
Enter passphrase (empty for no passphrase):
Enter same passphrase again:
Your identification has been saved in /home/username/.ssh/id_rsa.
Your public key has been saved in /home/username/.ssh/id_rsa.pub.
The key fingerprint is:
ef:87:b5:b1:4d:7e:69:95:3f:62:f5:0d:c0:7b:f1:5e username@hostname
The key's randomart image is:
+--[ RSA 2048]----+
|                 |
|                 |
|           .     |
|            o .  |
|        S    o o.|
|         .  + +oE|
|          .o O.oB|
|         .. +oo+*|
|          ... o..|
+-----------------+
```

When prompted, enter a passphrase. If you already have key pairs saved with the default names, you should enter a different file name for the new keys to avoid overwriting existing key pairs. More details on best practices can be found [ here](https://docs.alliancecan.ca/SSH_Keys#Best_practices_for_key_pairs " here"){.wikilink}.

## Creating a key pair backed by a hardware security key {#creating_a_key_pair_backed_by_a_hardware_security_key}

Some sites now support the use of SSH keys backed by a hardware security key (e.g. YubiKey). If you need one of these keys, you can generate it with the `ssh-keygen` command:

``` console
[name@yourLaptop]$  ssh-keygen -t ecdsa-sk
```

The output will be similar to the following:

``` console

Generating public/private ecdsa-sk key pair.
You may need to touch your authenticator to authorize key generation.
Enter file in which to save the key (/home/username/.ssh/id_ecdsa_sk):
Enter passphrase (empty for no passphrase): 
Enter same passphrase again: 
Your identification has been saved in /home/username/.ssh/id_ecdsa_sk
Your public key has been saved in /home/username/.ssh/id_ecdsa_sk.pub
The key fingerprint is:
SHA256:P051NAesYSxF7NruGLfnyAFMUBmGLwCaSRiXDwUY6Ts username@hostname
The key's randomart image is:
+-[ECDSA-SK 256]--+
|o*++o.  .o+Bo..  |
|+oo+  . .oo = .. |
|. +o   . ..+ oo .|
| .  .   .o. o. o |
|  .     S.oo. .  |
| E       ..o..   |
|  .       =.o    |
|         o *.+.  |
|          o.=o.  |
+----[SHA256]-----+
```

You will be prompted to both enter a passphrase and activate a hardware security key as part of the key creation process.

# Installing the public part of the key {#installing_the_public_part_of_the_key}

## Installing via CCDB {#installing_via_ccdb}

We encourage all users to leverage the new CCDB feature to install their SSH public key. This will make the key available to all our clusters. Grab the content of your public key (called *id_rsa.pub* in the above case) and upload it to CCDB as per step 3 of [these instructions](https://docs.alliancecan.ca/SSH_Keys#Using_CCDB "these instructions"){.wikilink}.

The simplest, safest way to install a key to a remote system is by using the `ssh-copy-id` command:

``` console
ssh-copy-id -i ~/.ssh/mynewkey.pub graham.computecanada.ca
```

This assumes that the new keypair is named \"mynewkey\" and \"mynewkey.pub\", and that your username on the remote machine is the same as your local username.

If necessary, you can do this \"manually\" - in fact, `ssh-copy-id` isn\'t doing anything very magical. It\'s simply connecting to the remote machine, and placing the public key into `.ssh/authorized_keys` in your /home directory there. The main benefit from using `ssh-copy-id` is that it will create files and directories if necessary, and will ensure that the permissions on them are correct. You can do it entirely yourself by copying the public key file to the remote server, then:

``` bash
mkdir ~/.ssh
cat id_rsa.pub >> ~/.ssh/authorized_keys
chmod --recursive go-rwx ~/.ssh
chmod go-w ~
```

SSH is picky about permissions, on both the client and the server. SSH will fail if the following conditions are not met:

- The private key file must not be accessible to others. `chmod go-rwx id_rsa`
- Your remote /home directory must not be writable by others. `chmod go-w ~`
- Same for your remote \~/.ssh and \~/.ssh/authorized_keys `chmod --recursive go-rwx ~/.ssh`.

Note that debugging the remote conditions may not be obvious without the help of the remote machine\'s system administrators.

# Connecting using a key pair {#connecting_using_a_key_pair}

<li>

Finally, test the new key by sshing to the remote machine from the local machine with

``` console
[name@yourLaptop]$ ssh -i /path/to/your/privatekey USERNAME@ADDRESS
```

where

:\*`/path/to/your/privatekey` specifies your private key file, e.g. `/home/ubuntu/.ssh/id_rsa`;

:\*`USERNAME` is the user name on the remote machine;

:\*`ADDRESS` is the address of the remote machine.

If you have administrative access on the server and created the account for other users, they should test the connection out themselves and not disclose their private key.

</li>
</ol>

# Using ssh-agent {#using_ssh_agent}

Having successfully created a key pair and installed the public key on a cluster, you can now log in using the key pair. While this is a better solution than using a password to connect to our clusters, it still requires you to type in a passphrase, needed to unlock your private key, every time that you want to log in to a cluster. There is however the `ssh-agent` program, which stores your private key in memory on your local computer and provides it whenever another program on this computer needs it for authentification. This means that you only need to unlock the private key once, after which you can log in to a remote cluster many times without having to type in the passphrase again.

You can start the `ssh-agent` program using the command

After you have started the `ssh-agent`, which will run in the background while you are logged in at your local computer, you can add your key pair to those managed by the agent using the command

Assuming you installed your key pair in one of the standard locations, the `ssh-add` command should be able to find it, though if necessary you can explicitly add the full path to the private key as an argument to `ssh-add`. Using the `ssh-add -l` option will show which private keys currently accessible to the `ssh-agent`.

While using `ssh-agent` will allow automatically negotiate the key exchange between your personal computer and the cluster, if you need to use your private key on the cluster itself, for example when interacting with a remote GitHub repository, you will need to enable *agent forwarding*. To enable this on the [Béluga](https://docs.alliancecan.ca/Béluga/en "Béluga"){.wikilink} cluster, you can add the following lines to your `$HOME/.ssh/config` file on your personal computer,

Note that you should never use the line `Host *` for agent forwarding in your SSH configuration file.

## Installing locally {#installing_locally}

Note that many contemporary Linux distributions as well as macOS now offer graphical \"keychain managers\" that can easily be configured to also manage your SSH key pair, so that logging in on your local computer is enough to store the private key in memory and have the operating system automatically provide it to the SSH client during login on a remote cluster. You will then be able to log in to our clusters without ever typing in any kind of passphrase.
