---
title: "Connecting with PuTTY/en"
url: "https://docs.alliancecan.ca/wiki/Connecting_with_PuTTY/en"
category: "General"
last_modified: "2023-04-24T18:42:53Z"
page_id: 1658
display_title: "Connecting with PuTTY"
---

`<languages />`{=html}

![ Enter hostname or IP address (Click for larger image)](https://docs.alliancecan.ca/Putty_basic.png " Enter hostname or IP address (Click for larger image)"){width="400"} ![ Specify username to use when connecting; this is optional as one can type it when connecting (Click for larger image)](https://docs.alliancecan.ca/Putty_username.png " Specify username to use when connecting; this is optional as one can type it when connecting (Click for larger image)"){width="400"} ![ Enable X11 forwarding (Click for larger image)](https://docs.alliancecan.ca/Putty_X11_forwarding.png " Enable X11 forwarding (Click for larger image)"){width="400"} ![ Specifying an SSH key (Click for larger image)](https://docs.alliancecan.ca/Putty_ssh_key.png " Specifying an SSH key (Click for larger image)"){width="400"}

Start up [PuTTY](http://www.chiark.greenend.org.uk/~sgtatham/putty/) and enter the host name or IP address of the machine you wish to connect to. You may also save a collection of settings by entering a session name in the `<i>`{=html}Save Sessions`</i>`{=html} text box and clicking the `<i>`{=html}Save`</i>`{=html} button. You can set the username to use when logging into a particular host under the `<i>`{=html}Connection-\>Data`</i>`{=html} section in the `<i>`{=html}Auto-login username`</i>`{=html} text box to saving typing the username when connecting.

# X11 forwarding {#x11_forwarding}

If working with graphical-based programs, X11 forwarding should be enabled. To do this, go to `<i>`{=html}Connection-\>SSH-\>X11`</i>`{=html} and check the `<i>`{=html}Enable X11 forwarding`</i>`{=html} checkbox. To use X11 forwarding one must install an X window server such as [Xming](http://www.straightrunning.com/xmingnotes/) or, for the recent versions of Windows, [VcXsrv](https://sourceforge.net/projects/vcxsrv/). The X window server should be actually started prior to connecting with SSH. Test that X11 forwarding is working by opening a PuTTY session and running a simple GUI-based program, such as typing the command `xclock`. If you see a popup window with a clock, X11 forwarding should be working.

# Using a key pair {#using_a_key_pair}

To set the private key putty uses when connecting to a machine go to Connection-\>SSH-\>Auth and clicking the `<i>`{=html}Browse`</i>`{=html} button to find the private key file to use. Putty uses files with a `<i>`{=html}.ppk`</i>`{=html} suffix, which are generated using PuTTYGen (see [Generating SSH keys in Windows](https://docs.alliancecan.ca/Generating_SSH_keys_in_Windows "Generating SSH keys in Windows"){.wikilink} for instructions on how to create such a key). In newer versions of Putty, you need to click the \"+\" sign next to `<i>`{=html}Auth`</i>`{=html} and then select `<i>`{=html}Credentials`</i>`{=html} to be able to browse for the `<i>`{=html}Private key file for authentication`</i>`{=html}. Note that the additional fields in that newer interface, i.e. `<i>`{=html}Certificate to use`</i>`{=html} and `<i>`{=html}Plugin to provide authentication response`</i>`{=html}, should be left blank.
