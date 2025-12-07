---
title: "VNC/en"
url: "https://docs.alliancecan.ca/wiki/VNC/en"
category: "General"
last_modified: "2025-11-28T18:39:32Z"
page_id: 8518
display_title: "VNC"
---

`<languages/>`{=html}

![MATLAB running via VNC.](https://docs.alliancecan.ca/Matlab-vnc.png "MATLAB running via VNC."){width="400"}

To remotely start the graphical user interface (gui) of a program, X11 forwarding over [SSH](https://docs.alliancecan.ca/SSH "wikilink") is commonly used. However the performance of this approach is often too slow to perform smooth complex graphics rotations. A much better alternative is to use [VNC](https://en.wikipedia.org/wiki/Virtual_Network_Computing) to connect to a remote desktop.

# Setup

To begin a VNC client will need to be installed on your desktop. A TigerVNC package is available for Windows, MacOS and most Linux distributions. The following shows how to download, install and configure TigerVNC securely for each of these systems.

## Windows

Download and run the latest stable vncviewer64-x.y.z.exe version package installer from [the official download page](https://sourceforge.net/projects/tigervnc/files/stable/) for example `vncviewer64-1.15.0.exe` (as of April 2025). Make sure you download the viewer and not the server. To create secure tunnels from your desktop to the vncserver as described in the sections below, you will need to open a terminal window and run the SSH command. This may be done using PowerShell standard on Windows 10 since the 1809 update.

## MacOS

Install the latest stable DMG package by going to [the official download page](https://sourceforge.net/projects/tigervnc/files/stable/) and click the green `<b>`{=html}Download Latest Version`</b>`{=html} button for `TigerVNC-1.15.0.dmg` (as of April 2025). Once the download is complete double click the DMG file to open it. A TigerVNC Viewer icon should appear in a popup window along with a LICENSE.TXT and README.rst file. To complete the installation, drag the tigervnc icon that appears into the Applications folder and/or the lower [app dock](https://support.apple.com/en-ca/guide/mac-help/mh35859/mac). To remove the popup you will need to unmount the DMG file. To do this open a New Finder Window, verify `View->ShowSidebar` is selected, click the small up arrow beside `TigerVNC-1.15.0` in the left side menu and lastly close the finder window. If you are running macOS Monterey 12.2 and [TigerVNC crashes](https://github.com/TigerVNC/tigervnc/issues/1423) then you must upgrade to this latest version.

## Linux

First install TigerVNC viewer with the package manager for your Linux version:

  Linux Version             Install Command
  ------------------------- ----------------------------------------
  Debian, Ubuntu            `sudo apt-get install tigervnc-viewer`
  Fedora, CentOS, or RHEL   `sudo yum install tigervnc`
  Gentoo                    `emerge -av net-misc/tigervnc`

Next, start TigerVNC by either finding it in the applications menu or running `vncviewer` on the command line of your laptop.

# Connect

Now you need a VNC server to connect to, such as a temporary vncserver started on a cluster login or compute node as shown below.

## Login nodes {#login_nodes}

You may run lightweight applications (that do not require a gpu) within a remote VNC desktop on a cluster login node (memory and cputime limits apply). To do this, you must first connect to a cluster login node. Using nibi cluster to demonstrate :

`[``<b>`{=html}`laptop``</b>`{=html}`:~] ssh nibi.alliancecan.ca`

Next run `vncserver -list` to check if you have an old unused vncserver(s) still running on whichever nibi login node you get connected to. If you do then kill them off by running the following pkill command on

`[``<b>`{=html}`l4``</b>`{=html}`(login node):~] ``pkill Xvnc -u $USER`

1\) Now you may start your vncserver on the login node as shown here:

`[``<b>`{=html}`l4``</b>`{=html}`(login node):~] vncserver -idletimeout 86400`\
` Desktop 'TurboVNC: l4.nibi.sharcnet:1 (yourusername)' started on display ``<b>`{=html}`l4``</b>`{=html}`.nibi.sharcnet:1`\
` Starting applications specified in /cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/bin/xstartup.turbovnc`\
`Log file is ``<span style="Color:green">`{=html}`/home/yourusername/.vnc/``<b>`{=html}`l4``</b>`{=html}`.nibi.sharcnet:1.log``</span>`{=html}

Note that the vncserver command provided by StdEnv/2023 is based on [turbovnc](https://turbovnc.org). When starting a new vncserver on a login node `-idletimeout seconds` should be added as shown above. Doing so will ensure your `vncserver` eventually terminates (once S seconds has elapsed with no VNC viewer connections) should you forget to terminate your vncviewer session by clicking `System -> Log out` in the vnc desktop. The first time you start vncserver you will be required to set a password which can be [changed](https://docs.alliancecan.ca/wiki/VNC#Vncserver_password) later. The password will be required to remotely connect to your desktop with a vncclient (such as vncviewer). The same password will be required when making [multiple connections](https://docs.alliancecan.ca/wiki/VNC#Multiple_connections) assuming you started your vncserver by appending the additional `-alwaysshared` option.

2\) Now determine which port your vncserver is listening on (5901 for this example) by running grep on the log file:

`[``<b>`{=html}`l4``</b>`{=html}`(login node):~] grep -iE "\sport|kill" ``<span style="Color:green">`{=html}`/home/yourusername/.vnc/``<b>`{=html}`l4``</b>`{=html}`.nibi.sharcnet:1.log``</span>`{=html}\
`25/08/2025 15:16:20 Listening for VNC connections on TCP port ``<span style="Color:blue">`{=html}`5901``</span>`{=html}

Now you may exit the login node. The vncserver you started will continue running until the time limit you specified (with the -idletimeout option) is reached.

`[``<b>`{=html}`<span style="Color:green">`{=html}`l4``</span>`{=html}`</b>`{=html}`(login node):~]  exit`\
`[``<b>`{=html}`laptop``</b>`{=html}`:~]`

3\) On your desktop start a SSH tunnel. Doing this will forward an arbitrary port (5905 in this example) to the port your VNC server is listening on (5901 according to the above).

`[``<b>`{=html}`laptop``</b>`{=html}`:~] ssh nibi.computecanada.ca -L ``<span style="Color:red">`{=html}`5905``</span>`{=html}`:``<b>`{=html}`<span style="Color:green">`{=html}`l4``</span>`{=html}`</b>`{=html}`:``<span style="Color:blue">`{=html}`5901``</span>`{=html}

4\) Next also your desktop, either click the `<i>`{=html}TigerVNC Viewer`</i>`{=html} application icon and enter **localhost:`<span style="Color:red">`{=html}5905`</span>`{=html}** in the `<b>`{=html}VNC viewer: Connection details`</b>`{=html} popup window dialogue box that appears \*\* OR \*\* open another terminal window and specify the following on the command line then hit enter. With either approach you should next get a popup window requesting the VNC authentication password that you previously setup. After successfully entering the password your remote Desktop should immediately appear.

`[``<b>`{=html}`laptop``</b>`{=html}`:~] vncviewer localhost:``<span style="Color:red">`{=html}`5905``</span>`{=html}

Although there are no system time limits on the login nodes for processes, there are memory and cputime limits that any applications you run in your remote desktop will be subject to. If you require more memory, cpu resources, or gpu access for applications you run in your desktop OR for graphics acceleration, then use the following procedure to start your VNC server on a cluster compute node instead and you may request them accordingly as explained with the salloc command.

## Compute nodes {#compute_nodes}

If your program requires memory and/or cputime limits greater than those provided on a cluster login node(s), then connect to a cluster compute node using the `salloc` command, start a VNC server, and then start a secure tunnel to it (with suitable port forwarding) and connect to it from your desktop with a vncviewer. This approach will give you dedicated access to your vnc server on a compute node with full graphical desktop however by default it will not have hardware-accelerated OpenGL capabilities.

`<b>`{=html}1) Start a VNC server`</b>`{=html}

Before starting your VNC server, log into a cluster (such as nibi) and create a compute node allocation using the `salloc` command (24hr time limit applies). For example, to request an [interactive job](https://docs.alliancecan.ca/Running_jobs#Interactive_jobs "wikilink") using 4 CPUs and 16GB of memory you could use the command:

`[``<b>`{=html}`l4``</b>`{=html}`(login node):~] salloc --time=1:00:00 --cpus-per-task=4 --mem=16000 --account=def-piusername`\
`salloc: Pending job allocation 1149016`\
`salloc: job 1149016 queued and waiting for resources`\
`salloc: job 1149016 has been allocated resources`\
`salloc: Granted job allocation 1149016`\
`salloc: Waiting for resource configuration`\
`salloc: Nodes ``<b>`{=html}`c48``</b>`{=html}` are ready for job`\
`[``<b>`{=html}`c48``</b>`{=html}`(compute node):~]`

Once your interactive job has started, set this environment variable to avoid any repetitive desktop errors:

`[``<b>`{=html}`c48``</b>`{=html}`(compute node):~] export XDG_RUNTIME_DIR=${SLURM_TMPDIR}`

Then, start a VNC server with `vncserver` noting which compute node your job is running on (`<b>`{=html}c48`</b>`{=html} in this example). If unsure use the `hostname` command to check. The first time you do this you will be prompted to set a password for your VNC server **DO NOT LEAVE THIS BLANK** otherwise anyone could connect to it and gain access to the files in your account. You may change the password later using the `vncpasswd` command. Continuing with the example:

`[``<b>`{=html}`c48``</b>`{=html}`(compute node):~] vncserver`\
` Desktop 'TurboVNC: c48.nibi.sharcnet:1 (yourusername)' started on display ``<b>`{=html}`c48``</b>`{=html}`.nibi.sharcnet:1`\
` Starting applications specified in /cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/bin/xstartup.turbovnc`\
`Log file is ``<span style="Color:green">`{=html}`/home/yourusername/.vnc/``<b>`{=html}`c48``</b>`{=html}`.nibi.sharcnet:1.log``</span>`{=html}

Run the grep command on the log file to determine which port your VNC server is listening on:

`[``<b>`{=html}`c48``</b>`{=html}`(compute node):~] grep -iE "\sport|kill" ``<span style="Color:green">`{=html}`/home/yourusername/.vnc/``<b>`{=html}`c48``</b>`{=html}`.nibi.sharcnet:1.log``</span>`{=html}\
`26/08/2025 10:43:36 Listening for VNC connections on TCP port ``<span style="Color:blue">`{=html}`5901``</span>`{=html}

`<b>`{=html}2) Set up a SSH tunnel to the VNC server`</b>`{=html}

Once your VNC server has been started, you must create a secure \"bridge\" or \"tunnel\" from your laptop to the compute node where your vncserver is running (as determined in the previous step above). There are two types of tunnel commands that maybe used depending on which cluster you are using.

For all clusters (`<b>`{=html}except`</b>`{=html} nibi) the previously recommended form of the tunnel command `ssh username@clustername -L localforwardedport:``<span style="Color:orange">`{=html}`computenode``</span>`{=html}`:remotelisteningport` may continue to be used. As an example, if a vncserver is started on `rorqual` compute node `<b>`{=html}`<span style="Color:orange">`{=html}rc12509`</span>`{=html}`</b>`{=html} and the local port on your laptop to be forwarded is again `<span style="Color:red">`{=html}5905`</span>`{=html} the appropriate tunnel command becomes:

`[``<b>`{=html}`laptop``</b>`{=html}`:~] ssh username@rorqual.alliancecan.ca -L ``<span style="Color:red">`{=html}`5905``</span>`{=html}`:``<b>`{=html}`<span style="Color:orange">`{=html}`rc12509``</span>`{=html}`</b>`{=html}`:``<span style="Color:blue">`{=html}`5901``</span>`{=html}\
`Duo two-factor login for username`\
`Enter a passcode or select one of the following options:`\
`[rc12509(compute node):~] `

For nibi, a new form of the tunnel command `ssh -J username@clustername -L localforwardedport:``<span style="Color:orange">`{=html}`localhost``</span>`{=html}`:remotelisteningport computenode` must be used. In addition a SSH key pair must created on your laptop with the contents of the pub key entered into your `~/.ssh/authorized_keys` file on `nibi`. This approach will also work on any other cluster and so may eventually be preferred. Continuing with the above example, where `<b>`{=html}`<span style="Color:green">`{=html}c48`</span>`{=html}`</b>`{=html} is the compute node that you started your vncserver on, and `<span style="Color:red">`{=html}5905`</span>`{=html} is the local port on your laptop being forwarded, the tunnel command would be:

`[``<b>`{=html}`laptop``</b>`{=html}`:~] ssh -J username@nibi.alliancecan.ca -L ``<span style="Color:red">`{=html}`5905``</span>`{=html}`:``<span style="Color:orange">`{=html}`localhost``</span>`{=html}`:``<span style="Color:blue">`{=html}`5901``</span>`{=html}` ``<b>`{=html}`<span style="Color:green">`{=html}`c48``<span>`{=html}`</b>`{=html}\
`Duo two-factor login for username`\
`Enter a passcode or select one of the following options:`\
`[c48(compute node):~]`

If you exit the node that your tunnel is connected to, you will no longer be able to connect to the VNC server with vncviewer. However since your vncserver will continue running, so you may regain access to it by simply starting a new tunnel. For more information about tunnels see [SSH tunnel](https://docs.alliancecan.ca/SSH_tunnelling "wikilink").

`<b>`{=html}3) Connect to the VNC server`</b>`{=html}

If you have a Linux desktop, open a new local terminal window and tell your VNC client to connect to **localhost:port**. The following example uses the TigerVNC `vncviewer` command to connect to the running VNC server on cdr768. You will be prompted for the VNC password that you set up earlier in order to connect.

`[``<b>`{=html}`laptop``</b>`{=html}`:~]$ vncviewer localhost:``<span style="Color:red">`{=html}`5905``</span>`{=html}\
` TigerVNC viewer v1.15.0`\
`Built on: 2025-02-16 03:59`\
`Copyright (C) 1999-2025 TigerVNC team and many others (see README.rst)`\
`See `[`https://www.tigervnc.org`](https://www.tigervnc.org)` for information on TigerVNC.`\
` Tue Aug 26 10:59:59 2025`\
`DecodeManager: Detected 12 CPU core(s)`\
`DecodeManager: Creating 4 decoder thread(s)`\
`CConn:       Connected to host localhost port 5905`\
`CConnection: Server supports RFB protocol version 3.8`\
`CConnection: Using RFB protocol version 3.8`\
`CConnection: Choosing security type VeNCrypt(19)`\
`CVeNCrypt:   Choosing security type TLSVnc (258)`\
` Tue Aug 26 11:00:03 2025`\
`CConn:       Using pixel format depth 24 (32bpp) little-endian rgb888`\
`CConnection: Enabling continuous updates`

If you are on a Mac or Windows desktop (not a linux distro) then instead of running the `vncviewer` from the command line, you may click the `<i>`{=html}TigerVNC Viewer`</i>`{=html} application icon and enter your **localhost:port** information as shown here: [400px\|thumb**Mac Tiger VNC Viewer Connection Details Dialogue Box**](https://docs.alliancecan.ca/File:VNCviewerConnect3.png "wikilink"). As a side note, the default VNC port assumed by `<i>`{=html}TigerVNC Viewer`</I>`{=html} is 5900, therefore if you specified 5900 as the local port to be forwarded when you started your SSH tunnel, then you could simply specify **localhost**. Windows users however may find they cannot set up an SSH tunnel on local port 5900 in the first place.

Once `vncviewer` connects you will be presented with a [Linux MATE desktop](https://mate-desktop.org/). To launch a terminal, click on the top menu on \"Applications -\> System Tools -\> MATE Terminal\". You may also add a shortcut to the top menu by right-clicking on \"MATE Terminal\" and by clicking on \"Add this launcher to panel\". Finally, to launch a program, invoke the command as you would normally within a `bash` session, for example `xclock`. To start a more complicated program like MATLAB, load the module and then run the `matlab` command.

# More information {#more_information}

## Vncserver password {#vncserver_password}

To reset your VNC server password, use the `vncpasswd` command:

``` bash
[gra-login1:~] vncpasswd
Password:
Verify:
Would you like to enter a view-only password (y/n)? n
```

Optionally you can completely remove your VNC configuration (including your password) by deleting your `~/.vnc` directory. The next time you run `vncserver` you will be prompted to set a new password.

## Killing vncserver {#killing_vncserver}

If a running vncserver is no longer needed, terminate it with `vncserver -kill :DISPLAY#` for example:

`[gra-login1:~] vncserver -list | grep -v ^$`\
`TurboVNC sessions:`\
`X DISPLAY #    PROCESS ID  NOVNC PROCESS ID`\
`:``<span style="color:red">`{=html}`44``</span>`{=html}`         27644`\
` [gra-login1:~] vncserver -kill :``<span style="color:red">`{=html}`44``</span>`{=html}\
`Killing Xvnc process ID 27644`

If you have multiple vncservers running on a node, you may kill them ALL instantly by running:

`[gra-login1:~] pkill Xvnc -u $USER`

## Multiple connections {#multiple_connections}

All vncserver(s) running under your username (on a login or compute node) can be displayed with `vncserver -list`. If a vncserver was started with the additional `-AlwaysShared` option then multiple connections to it can be made by establishing a new tunnel and vncviewer from any remote location. For example:

`[``<b>`{=html}`l4``</b>`{=html}`(login node):~] vncserver -idletimeout 86400 -alwaysshared | grep -v ^$`\
`Desktop 'TurboVNC: l4.nibi.sharcnet:1 (yourusername)' started on display ``<b>`{=html}`l4``</b>`{=html}`.nibi.sharcnet:1`\
`Starting applications specified in /cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/bin/xstartup.turbovnc`\
`Log file is /home/yourusername/.vnc/``<b>`{=html}`l4``</b>`{=html}`.nibi.sharcnet:1.log`

Thus one could start a vncserver running while at the office and then go home, establish new tunnels to the login or compute node where the vncserver is still running, and re-connect again with vncviewer to access the same desktop and seamlessly continue working. If however your vncserver was not started with `vncserver -AlwaysShared` then only one vncviewer connection will be possible and you would need to close down all applications running in the desktop then shut down your vncserver, all before going home. Then later once home you would need to restart a whole new desktop from scratch and all applications just to finally continue working.

## Failures to connect {#failures_to_connect}

Repeated failing attempts to establish a new vncserver/vncviewer connection may be due to an old SSH tunnel still running on your desktop tying up ports. To identify and kill any such tunnels, open a terminal window on your desktop and run `ps ux | grep ssh` followed by `kill PID`.

## Unlock screensaver {#unlock_screensaver}

If your VNC screensaver times out and requests a password, enter your cluster account password to unlock it (not your vncserver password). If you are running the MATE desktop and the screensaver will not unlock, try running `killall -9 .mate-screensaver`. This should no longer be a problem on our clusters as the VNC screensaver has been disabled.

## Cannot log in {#cannot_log_in}

The procedure to log in to gra-vdi.alliancecan.ca is a two step process:

1\)

`username`\
`Enter your (ccdb) password`

2\)

`username`\
`Enter your Duo two-factor MFA passcode`

If you enter the wrong username/password for 1) you will still be prompted by 2). If you then send your username/passcode then you will receive a message that says `<b>`{=html}Success, Logging you in\...`</b>`{=html} and be returned to the log in screen of 1). The solution is to try again being sure to enter your correct username/password combination. If you cannot recall your CCDB password, visit [here](https://ccdb.alliancecan.ca/security/forgot) to reset it, assuming your account it not pending renewal by your PI.

## OpenGL graphics {#opengl_graphics}

To run a graphics-based program that uses hardware-based accelerated OpenGL, a couple of changes will be required in the above `<i>`{=html}Compute nodes`</i>`{=html} section.

First, the `salloc` command must be modified to request a GPU node. If this is not done, the program will fall back to using software-based rendering on CPUs, which is relatively much slower. To request the first GPU node that brcomes available (and in turn minimize your queue wait time if the cluster has multiple GPU node types) simply specify:

`[``<b>`{=html}`l4``</b>`{=html}`(login node):~] salloc --time=1:00:00 --cpus-per-task=4 --gpus-per-node=1 --mem=16000 --account=def-piname`

If however the cluster you are using has multiple node types, where one is known to provide good graphics acceleration such as a node with a `<b>`{=html}t4`</b>`{=html} gpu, then specify :

`[``<b>`{=html}`l4``</b>`{=html}`(login node):~] salloc --time=1:00:00 --cpus-per-task=4 --gpus-per-node=t4:1 --mem=16000 --account=def-piname`

Second, `vglrun` will probably need to be added just before the name of your `PROGRAM` on the command line of your VNC desktop terminal window. For example :

` [``<b>`{=html}`c48``</b>`{=html}`(compute node):~] vglrun -d egl PROGRAM`

Then `vglrun` sets some extra environment variables to ensure your program will use correct virtualgl libraries. If however your `PROGRAM` has already been patched to use the current cvmfs standard environment doing so will not be required.

## Portal alternatives {#portal_alternatives}

If you experience graphics issues when using VNC as described above, try instead using [OpenOnDemand](https://ondemand.sharcnet.ca/) on the `<b>`{=html}Nibi`</b>`{=html} cluster or [JupyterHub](https://jupyterhub.rorqual.alliancecan.ca) on the `<b>`{=html}Rorqual`</b>`{=html} cluster. Both systems offer an automated modern desktop VDI web interface GUI experience that is designed for ease of use with improved hardware performance and software support.
