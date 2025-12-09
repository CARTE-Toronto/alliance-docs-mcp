---
title: "Transferring data/en"
url: "https://docs.alliancecan.ca/wiki/Transferring_data/en"
category: "General"
last_modified: "2025-08-29T12:50:27Z"
page_id: 2144
display_title: "Transferring data"
---

Please use data transfer nodes, also called data mover nodes, instead of login nodes whenever you are transferring data to and from our clusters. If a data transfer node is available, its URL will be given near the top of the main page for each cluster, which you can find in the sidebar on the left.

Globus automatically uses data transfer nodes.

==To and from your personal computer==
You will need software that supports secure transfer of files between your computer and our machines. The commands scp and sftp can be used in a command-line environment on Linux or Mac OS X computers. On Microsoft Windows platforms, MobaXterm offers both a graphical file transfer function and a command-line interface via SSH, while WinSCP is another free program that supports file transfer. Setting up a connection to a machine using SSH keys with WinSCP can be done by following the steps in this link. PuTTY comes with pscp and psftp which are essentially the same as the Linux and Mac command line programs.

If it takes more than one minute to move your files to or from our servers, we recommend you install and try Globus Personal Connect. Globus transfers can be set up and will run in the background.

==Between resources==
Globus is the preferred tool for transferring data between systems, and if it can be used, it should.

However, other common tools can also be found for transferring data both inside and outside of our systems, including
*  SFTP
*  SCP or Secure Copy Protocol
*  rsync

Note: If you want to transfer files between another of our clusters and Trillium use the SSH agent forwarding flag -A when logging into another cluster. For example, to copy files to Trillium from Fir, use:

ssh -A USERNAME@fir.alliancecan.ca

then perform the copy:

[USERNAME@fir2 ~]$ scp file USERNAME@trillium.alliancecan.ca:

==From the World Wide Web==
The standard tool for downloading data from websites is wget. Another often used is curl. Their similarities and differences are compared in several places such as this StackExchange article or here.  While the focus here is transferring data on Alliance Linux systems this tutorial also addresses Mac and Windows machines.   Both wget and curl can resume interrupted downloads by rerunning them with the -c and -C - command line options respectively. When getting data from various cloud services such as Google cloud storage, Google Drive and Google Photos, consider using the rclone tool instead.  All of these tools (wget, curl, rclone) are available on every Alliance cluster by default (without loading a module).  For a detailed listing of command line options check the man page for each tool or run them with --help or simply -h on the cluster.

==Synchronizing files==
To synchronize or sync files (or directories) stored in two different locations means to ensure that the two copies are the same. Here are several different ways to do this.

===Globus transfer===
We find Globus usually gives the best performance and reliability.

Normally when a Globus transfer is initiated it will overwrite the files on the destination with the files from the source, which means all of the files on the source will be transferred. If some of the files may already exist on the destination and need not be transferred if they match, you should go to the Transfer & Timer Options shown in the screenshot and choose to sync instead.

You may choose how Globus decides which files to transfer:

Their checksums are different    	This is the slowest option but most accurate. This will catch changes or errors that result in the same size of file, but with different contents.
File doesn't exist on destination	This will only transfer files that have been created since the last sync. Useful if you are incrementally creating files.
File size is different           	A quick test. If the file size has changed then its contents must have changed, and it will be re-transferred.
Modification time is newer       	This will check the file's recorded modification time and only transfer the file if it is newer on the source than the destination. If you want to depend on this, it is important to check the preserve source file modification times option when initiating a Globus transfer.

For more information about Globus please see Globus.

===Rsync===
Rsync is a popular tool for ensuring that two separate datasets are the same but can be quite slow if there are a lot of files or there is a lot of latency between the two sites, i.e. they are geographically apart or on different networks. Running rsync will check the modification time and size of each file, and will only transfer the file if one or the other does not match. If you expect modification times not to match on the two systems, you can use the -c option, which will compute checksums at the source and destination, and transfer only if the checksums do not match.

When transferring files into the /project file systems, do not use -p and -g flags since the quotas in /project are enforced based on group ownership, and thus preserving the group ownership will lead to the  Disk quota exceeded error message.  Since -a includes -p and -g by default, the --no-g --no-p options should be added, like so

where LOCALNAME can be a directory or file preceded by its path location and somedir will be created if it doesn't exist.  The -z option compresses files (not in the default file suffixes --skip-compress list) and requires additional cpu resources while the -h option makes transferred file sizes human readable.  If you are transferring very large files add the --partial option so interrupted transfers maybe restarted:

The --progress option will display the percent progress of each file as its transferred.  If you are transferring very many smaller files, then it maybe more desirable to display a single progress bar that represents the transfer progress of all files:
progress2 LOCALNAME someuser@nibi.alliancecan.ca:projects/def-professor/someuser/somedir/}}
The above rsync examples all involve transfers from a local system into a project directory on a remote system.  Rsync transfers from a remote system into a project directory on a local system work in much the same way, for example:

where REMOTENAME can be a directory or file preceded by its path location and somedir will be created if it doesn't already exist.  In its simplest incarnation rsync can also be used locally within a single system to transfer a directory or file (from home or scratch) into project by dropping the cluster name:

where somedir will be created if it doesn't already exist before copying LOCALNAME into it.  For comparison purposes, the copy command can similarely be used to transfer LOCALNAME from home to project by doing:
"mode,timestamps" LOCALNAME ~/projects/def-professor/someuser/somedir/}}
however unlike rsync, if LOCALNAME is a directory, it will be renamed to somedir if somedir does not exist.

===Using checksums to check if files match===
If Globus is unavailable between the two systems being synchronized and Rsync is taking too long, then you can use a  checksum utility on both systems to determine if the files match. In this example we use sha1sum.

 xargs -0 sha1sum  tee checksum-result.log
}}

This command will create a new file called checksum-result.log in the current directory; the file will contain all of the checksums for the files in /home/username/. It will also print out all of the checksums to the screen as it goes. If you have a lot of files or very large files you may want to run this command in the background, in a screen or tmux session; anything that allows it to continue if your SSH connection times out.

After you run it on both systems, you can use the diff utility to find files that don't match.

It is possible that the find command will crawl through the directories in a different order, resulting in a lot of false differences so you may need to run sort on both files before running diff such as:

==SFTP==
SFTP (Secure File Transfer Protocol) uses the SSH protocol to transfer files between machines which encrypts data being transferred.

For example, you can connect to a remote machine at ADDRESS as user USERNAME with SFTP to transfer files like so:

[name@server]$ sftp USERNAME@ADDRESS
The authenticity of host 'ADDRESS (###.###.###.##)' can't be established.
RSA key fingerprint is ##:##:##:##:##:##:##:##:##:##:##:##:##:##:##:##.
Are you sure you want to continue connecting (yes/no)? yes
Warning: Permanently added 'ADDRESS,###.###.###.##' (RSA) to the list of known hosts.
USERNAME@ADDRESS's password:
Connected to ADDRESS.
sftp>

or using an SSH Key for authentication using the -i option

[name@server]$ sftp -i /home/name/.ssh/id_rsa USERNAME@ADDRESS
Connected to ADDRESS.
sftp>

which returns the sftp> prompt where commands to transfer files can be issued. To get a list of commands available to use at the sftp prompt enter the help command.

There are also a number of graphical programs available for Windows, Linux and Mac OS, such as WinSCP and MobaXterm (Windows), filezilla (Windows, Mac, and Linux), and cyberduck (Mac and Windows).

==SCP==

SCP stands for Secure Copy Protocol. Like SFTP it uses the SSH protocol to encrypt data being transferred. It does not support synchronization like Globus or rsync. Some examples of the most common use of SCP include

which will copy the file foo.txt from the current directory on my local computer to the directory $HOME/work on the cluster Rorqual. To copy a file, output.dat from my project space on the cluster Fir to my local computer I can use a command like

Many other examples of the use of SCP are shown here. Note that you always execute this scp command on your local computer, not the remote cluster - the SCP connection, regardless of whether you are transferring data to or from the remote cluster, should always be initiated from your local computer.

SCP supports the option -r to recursively transfer a set of directories and files. We recommend against using scp -r to transfer data into /project because the setgid bit is turned off in the created directories, which may lead to Disk quota exceeded or similar errors if files are later created there (see  Disk quota exceeded error on /project filesystems).

***Note*** if you chose a custom SSH key name, i.e. something other than the default names: id_dsa, id_ecdsa, id_ed25519 and id_rsa, you will need to use the -i option of scp and specify the path to your private key before the file paths via

==Prevention and Troubleshooting==

===Unable to read data===

Before initiating any transfer, make sure you can read all the contents of the directories you would like to transfer. On a Linux system, the following command lists all items not readable to you:

===Unable to write new data===

* Double-check the storage usage and make sure enough space and enough files are available.
** On some clusters, the filesystem automatically compresses your files and reports the space usage by the disk usage of the compressed data. On other clusters, the space usage is reported by the apparent size of your files. Therefore, 1 TB of compressed data on one cluster may become 2 TB of data on the next cluster.
** Before transferring a dataset, it is possible to get its apparent size with the option -b of the du command:

* Double-check the filesystem permissions and make sure you have the write permission at the location where you are trying to transfer new files.