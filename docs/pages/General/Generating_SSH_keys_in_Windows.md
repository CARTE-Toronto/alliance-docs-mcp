---
title: "Generating SSH keys in Windows/en"
url: "https://docs.alliancecan.ca/wiki/Generating_SSH_keys_in_Windows/en"
category: "General"
last_modified: "2023-01-31T20:50:36Z"
page_id: 1802
display_title: "Generating SSH keys in Windows"
---

`<languages />`{=html}

*Parent page: [SSH](https://docs.alliancecan.ca/SSH "SSH"){.wikilink}*

![ PuTTYgen before generating a key (Click for larger image)](https://docs.alliancecan.ca/Puttygen1.png " PuTTYgen before generating a key (Click for larger image)"){width="400"} ![ PuTTYgen after generating a key (Click for larger image)](https://docs.alliancecan.ca/Puttygen2.png " PuTTYgen after generating a key (Click for larger image)"){width="400"}

# Generating a key pair {#generating_a_key_pair}

The process of generating a key is nearly the same whether you are using PuTTY or MobaXTerm.

- With MobaXTerm, go to the menu item Tools-\>MobaKeyGen (SSH key generator)
- With PuTTY, run the PuTTYGen executable.

Both of these methods will cause a window to be displayed which can be used to generate a new key or to load an existing key. The PuTTY window is illustrated at right. The MobaXTerm window looks almost exactly the same.

1.  For \"Type of key to generate\" select \"Ed25519\". (Type \"RSA\" is also acceptable, but set the \"Number of bits\" to 2048 or greater.)
2.  Click the \"Generate\" button. You will then be asked to move your mouse around to generate random data to be used to create the key.
3.  Enter a passphrase for your key. Remember this passphrase, you will need it every time you reload PuTTY or MobaXTerm to use this key pair.
4.  Click \"Save private key\" and choose a meaningful file name; the extention `.ppk` is added to the file name. (e.g. compute_canada.ppk).
5.  Click \"Save public key\". It is conventional to save the public key with the same name as the private key, but here, the extension is `.pub`.

# Installing the public part of the key pair {#installing_the_public_part_of_the_key_pair}

## Installing via CCDB {#installing_via_ccdb}

We encourage you to register your SSH public key with the CCDB. This will let you to use it to log in to any of our HPC clusters. Copy the contents of the box titled \"Public key for pasting into OpenSSH \...\" and paste it into the box at [CCDB -\> Manage SSH Keys](https://ccdb.computecanada.ca/ssh_authorized_keys). For more about this, see [ SSH Keys: Using CCDB](https://docs.alliancecan.ca/SSH_Keys#Using_CCDB " SSH Keys: Using CCDB"){.wikilink}.

## Installing locally {#installing_locally}

If for some reason you do not want to use the CCDB method, you may upload your public key onto `<em>`{=html}each`</em>`{=html} cluster as follows:

1.  Copy the contents of the box titled \"Public key for pasting into OpenSSH \...\" and paste it as a single line at the end of `/home/USERNAME/.ssh/authorized_keys` on the cluster you wish to connect to.
2.  Ensure the permissions and ownership of the `~/.ssh` directory and files therein are correct, as described in [these instructions](https://docs.alliancecan.ca/Using_SSH_keys_in_Linux#Installing_locally "these instructions"){.wikilink}.

You may also use `ssh-copy-id` for this purpose, if it is available on your personal computer.

# Connecting using a key pair {#connecting_using_a_key_pair}

Test the new key by connecting to the server using SSH. See [ connecting with PuTTY using a key pair](https://docs.alliancecan.ca/Connecting_with_PuTTY#Using_a_Key_Pair " connecting with PuTTY using a key pair"){.wikilink}; [ connecting with MobaXTerm using a key pair](https://docs.alliancecan.ca/Connecting_with_MobaXTerm#Using_a_Key_Pair " connecting with MobaXTerm using a key pair"){.wikilink}; or [connecting with WinSCP](https://winscp.net/eng/docs/ui_login_authentication).

Key generation and usage with PuTTY is demonstrated in this video : [Easily setup PuTTY SSH keys for passwordless logins using Pageant](https://www.youtube.com/watch?v=2nkAQ9M6ZF8).

# Converting an OpenStack key {#converting_an_openstack_key}

When a key is created on [OpenStack](https://docs.alliancecan.ca/Managing_your_cloud_resources_with_OpenStack "OpenStack"){.wikilink} you obtain a key with a \".pem\" extension. This key can be converted to a format used by PuTTY by clicking the \"Load\" button in PuTTYGen. Then select the \"All Files (\*.\*)\" filter, select the \".pem\" file you downloaded from OpenStack, and click \"Open\". You should also add a \"Key passphrase\" at this point to use when accessing your private key and then click \"Save private key\".

This private key can be used with PuTTY to connect to a VM created with OpenStack. For more about this, see \"Launching a VM\" on the [Cloud Quick Start](https://docs.alliancecan.ca/Cloud_Quick_Start "Cloud Quick Start"){.wikilink} page.
