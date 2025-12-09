---
title: "Connecting with MobaXTerm/en"
url: "https://docs.alliancecan.ca/wiki/Connecting_with_MobaXTerm/en"
category: "General"
last_modified: "2019-03-21T14:12:37Z"
page_id: 1665
display_title: "Connecting with MobaXTerm"
---

`<languages />`{=html} ![ Creating an SSH session (Click for larger image)](https://docs.alliancecan.ca/MobaXterm_basic.png " Creating an SSH session (Click for larger image)"){width="600"} ![ Connected to a remote host (Click for larger image)](https://docs.alliancecan.ca/MobaXterm_connected.png " Connected to a remote host (Click for larger image)"){width="600"} ![ Enabling X11 Forwarding(Click for larger image)](https://docs.alliancecan.ca/MobaXterm_X11.png " Enabling X11 Forwarding(Click for larger image)"){width="600"} ![ Specifying a private key (Click for larger image)](https://docs.alliancecan.ca/MobaXterm_ssh_key.png " Specifying a private key (Click for larger image)"){width="600"}

Connecting with [MobaXterm](http://mobaxterm.mobatek.net/) works in basically the same way as PuTTY (see [Connecting with PuTTY](https://docs.alliancecan.ca/Connecting_with_PuTTY "Connecting with PuTTY"){.wikilink}) however, there is more functionality combined into MobaXterm than PuTTY. MobaXterm has a built-in SFTP client to transfer files as well as a built-in X11 server to allow you to run graphical programs remotely without the need to install a third-party X11 server. If you have already been using PuTTY and have saved sessions, MobaXterm will use these saved sessions so that you do not have to re-enter the settings.

To connect to a machine which you have not previously connected to using MobaXterm or PuTTY go to Sessions-\>New session, select an \"SSH\" session, type in the remote host address and your USERNAME (note you may need to check the \"Specify username\" check box). Then click \"OK\". MobaXterm will then save that session information you just entered for future connections, and also open an SSH connection to the specified host, which will then request your password. Once your password is entered successfully you will now have a terminal you can type commands at as well as an SFTP client in the left pane which you can use to view files on the remote machine as well as transfer files to and from the remote machine by dragging and dropping files.

# X11 Forwarding {#x11_forwarding}

To enable X11 forwarding to allow the use of graphical applications from the host machine:

1.  Ensure that X11 forwarding is enabled for a particular session by right clicking on the session and select \"Edit Session\". In the session settings window, select \"Advanced SSH settings\" and ensure that the \"X11-Forwarding\" checkbox is checked.
2.  Ensure that the Icon for the \"X server\" in the top right corner of the main window is green. If it isn\'t green that means that you do not currently have an X server running. To start, click on the red \"X\" icon.
3.  Test that X11 forwarding is working by opening the session by double-clicking the session on the \"Sessions\" pane on the left and entering your password. Then run a simple GUI-based program to test, such as typing the command `xclock`. If you see a popup window with a clock, X11 forwarding should be working.

# Using a Key Pair {#using_a_key_pair}

Right-click on the session in the left \"Sessions\" pane and select \"Edit Session\". In the session settings window, select \"Advanced SSH settings\" and check the \"Use private key\" checkbox. You can then click on the icon at the right of the text box to browse the file system and select a private key file to use. To create a key pair, see [Generating SSH keys in Windows](https://docs.alliancecan.ca/Generating_SSH_keys_in_Windows "Generating SSH keys in Windows"){.wikilink}.
