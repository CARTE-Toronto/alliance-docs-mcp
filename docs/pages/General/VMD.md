---
title: "VMD/en"
url: "https://docs.alliancecan.ca/wiki/VMD/en"
category: "General"
last_modified: "2025-08-29T13:37:02Z"
page_id: 10582
display_title: "VMD"
---

"VMD is a molecular visualization program for displaying, animating, and analyzing large biomolecular systems using 3-D graphics and built-in scripting."
The VMD web site is here.

== Using a pre-installed version ==

Connect to a cluster using VNC, ssh -X, or ssh -Y, in order to enable graphics.
We recommend using VNC for best performance.

To run the default version of VMD, currently 1.9.4a57, do:

This should open VMD graphical windows.

See Using modules for more on the module command, including how to find and use other pre-installed versions.

== Installing version 1.9.4 Alpha ==

1. Download the 1.9.4 LATEST ALPHA tar file from http://www.ks.uiuc.edu/, selecting the LINUX_64 version.  Free registration is required.

2. Copy the file to the home directory of the cluster you wish to use.

3. Unpack the file:

 tar xvf vmd-1.9.4*.opengl.tar.gz

4. Enter the newly-created directory:

 cd vmd-1.9.4*

5. Create two new directories to receive the program files:

 mkdir ~/vmd_install
 mkdir ~/vmd_library

5. Edit the configure file to read as follows, replacing each instance of your_user_name with your actual user name:

 # Directory where VMD startup script is installed, should be in users' paths.
 $install_bin_dir="/home/your_user_name/vmd_install";

 # Directory where VMD files and executables are installed
 $install_library_dir="/home/your_user_name/vmd_library";

6. Run configure and make:

 ./configure
 cd src
 make install

7. Add the resulting executable to your path:
 export PATH=~/vmd_install:$PATH

8. Use setrpaths.sh to modify the VMD executables so they use libraries from CVMFS:
 cd ~/vmd_library/
 setrpaths.sh  --path .

If you are using a Mac and getting a blank window, try running this:

 defaults write org.macosforge.xquartz.X11 enable_iglx -bool true

== Installing plugins ==

VMD has many plugins available.  You can install them in your own space.
The example that follows illustrates how to install the CaFE plugin,
from detailed instructions which can be found here:

 wget https://github.com/HuiLiuCode/CaFE_Plugin/archive/refs/heads/master.zip
 unzip master.zip
 cd CaFE_Plugin-master
 mv src cafe1.0
 mv cafe1.0 ~
 cd

Edit the .vmdrc file with your favorite editor (nano, vim, emacs etc.) and add this line:

 set auto_path [linsert $auto_path 0 {~/cafe1.0}]

After this, load the vmd module and any other modules which are required, such as namd and the CaFE plugin should be available.

== Links ==

* WestGrid webinars (in English)
** Molecular visualization with VMD
** Advanced VMD: Trajectories, movies, scripting