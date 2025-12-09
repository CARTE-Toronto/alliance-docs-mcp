---
title: "Transferring data/en"
url: "https://docs.alliancecan.ca/wiki/Transferring_data/en"
category: "General"
last_modified: "2025-08-29T12:50:27Z"
page_id: 2144
display_title: "Transferring data"
---

`<languages />`{=html}

Please use `<b>`{=html}data transfer nodes`</b>`{=html}, also called `<b>`{=html}data mover nodes`</b>`{=html}, instead of login nodes whenever you are transferring data to and from our clusters. If a data transfer node is available, its URL will be given near the top of the main page for each cluster, which you can find in the sidebar on the left.

[Globus](https://docs.alliancecan.ca/Globus "Globus"){.wikilink} automatically uses data transfer nodes.

## To and from your personal computer {#to_and_from_your_personal_computer}

You will need software that supports secure transfer of files between your computer and our machines. The commands `scp` and `sftp` can be used in a command-line environment on `<b>`{=html}Linux`</b>`{=html} or `<b>`{=html}Mac`</b>`{=html} OS X computers. On `<b>`{=html}Microsoft Windows`</b>`{=html} platforms, [MobaXterm](https://docs.computecanada.ca/wiki/Connecting_with_MobaXTerm/en) offers both a graphical file transfer function and a [command-line](https://docs.alliancecan.ca/Linux_introduction "command-line"){.wikilink} interface via [SSH](https://docs.alliancecan.ca/SSH "SSH"){.wikilink}, while [WinSCP](http://winscp.net/eng/index.php) is another free program that supports file transfer. Setting up a connection to a machine using SSH keys with WinSCP can be done by following the steps in this [link](https://www.exavault.com/blog/import-ssh-keys-winscp). [PuTTY](https://docs.computecanada.ca/wiki/Connecting_with_PuTTY/en) comes with `pscp` and `psftp` which are essentially the same as the Linux and Mac command line programs.

If it takes more than one minute to move your files to or from our servers, we recommend you install and try [Globus Personal Connect](https://docs.alliancecan.ca/Globus#Personal_Computers "Globus Personal Connect"){.wikilink}. [Globus](https://docs.alliancecan.ca/Globus "Globus"){.wikilink} transfers can be set up and will run in the background.

## Between resources {#between_resources}

[Globus](https://docs.alliancecan.ca/Globus "Globus"){.wikilink} is the preferred tool for transferring data between systems, and if it can be used, it should.

However, other common tools can also be found for transferring data both inside and outside of our systems, including

- [ SFTP](https://docs.alliancecan.ca/Transferring_data#SFTP " SFTP"){.wikilink}
- [ SCP](https://docs.alliancecan.ca/Transferring_data#SCP " SCP"){.wikilink} or Secure Copy Protocol
- [ rsync](https://docs.alliancecan.ca/Transferring_data#Rsync " rsync"){.wikilink}

Note: If you want to transfer files between another of our clusters and Trillium use the SSH agent forwarding flag `-A` when logging into another cluster. For example, to copy files to Trillium from Fir, use:

    ssh -A USERNAME@fir.alliancecan.ca

then perform the copy:

    [USERNAME@fir2 ~]$ scp file USERNAME@trillium.alliancecan.ca:

## From the World Wide Web {#from_the_world_wide_web}

The standard tool for downloading data from websites is [wget](https://en.wikipedia.org/wiki/Wget). Another often used is [curl](https://curl.haxx.se/). Their similarities and differences are compared in several places such as this StackExchange [article](https://unix.stackexchange.com/questions/47434/what-is-the-difference-between-curl-and-wget) or [here](https://draculaservers.com/tutorials/wget-and-curl-for-files/). While the focus here is transferring data on Alliance Linux systems this [tutorial](https://www.techtarget.com/searchnetworking/tutorial/Use-cURL-and-Wget-to-download-network-files-from-CLI) also addresses Mac and Windows machines. Both [wget](https://www.thegeekstuff.com/2009/09/the-ultimate-wget-download-guide-with-15-awesome-examples/) and [curl](https://www.thegeekstuff.com/2012/04/curl-examples/) can resume interrupted downloads by rerunning them with the [-c](https://www.cyberciti.biz/tips/wget-resume-broken-download.html) and [-C -](https://www.cyberciti.biz/faq/curl-command-resume-broken-download/) command line options respectively. When getting data from various cloud services such as Google cloud storage, Google Drive and Google Photos, consider using the [rclone](https://rclone.org/) tool instead. All of these tools (wget, curl, rclone) are available on every Alliance cluster by default (without loading a module). For a detailed listing of command line options check the man page for each tool or run them with `--help` or simply `-h` on the cluster.

## Synchronizing files {#synchronizing_files}

To synchronize or `<i>`{=html}sync`</i>`{=html} files (or directories) stored in two different locations means to ensure that the two copies are the same. Here are several different ways to do this.

### Globus transfer {#globus_transfer}

We find Globus usually gives the best performance and reliability.

Normally when a Globus transfer is initiated it will overwrite the files on the destination with the files from the source, which means all of the files on the source will be transferred. If some of the files may already exist on the destination and need not be transferred if they match, you should go to the `<i>`{=html}Transfer & Timer Options`</i>`{=html} shown in the screenshot and choose to `<i>`{=html}sync`</i>`{=html} instead.

![](Globus_Transfer_Sync_Options.png "Globus_Transfer_Sync_Options.png"){width="280"}

You may choose how Globus decides which files to transfer:

  ------------------------------------ -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Their checksums are different        This is the slowest option but most accurate. This will catch changes or errors that result in the same size of file, but with different contents.
  File doesn\'t exist on destination   This will only transfer files that have been created since the last sync. Useful if you are incrementally creating files.
  File size is different               A quick test. If the file size has changed then its contents must have changed, and it will be re-transferred.
  Modification time is newer           This will check the file\'s recorded modification time and only transfer the file if it is newer on the source than the destination. If you want to depend on this, it is important to check the `<i>`{=html}preserve source file modification times`</i>`{=html} option when initiating a Globus transfer.
  ------------------------------------ -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

For more information about Globus please see [Globus](https://docs.alliancecan.ca/Globus "Globus"){.wikilink}.

\

### Rsync

[Rsync](https://en.wikipedia.org/wiki/Rsync) is a popular tool for ensuring that two separate datasets are the same but can be quite slow if there are a lot of files or there is a lot of latency between the two sites, i.e. they are geographically apart or on different networks. Running `rsync` will check the modification time and size of each file, and will only transfer the file if one or the other does not match. If you expect modification times not to match on the two systems, you can use the `-c` option, which will compute checksums at the source and destination, and transfer only if the checksums do not match.

When transferring files into the `/project` file systems, do not use `-p` and `-g` flags since the quotas in `/project` are enforced based on group ownership, and thus preserving the group ownership will lead to the [ Disk quota exceeded](https://docs.alliancecan.ca/Frequently_Asked_Questions#Disk_quota_exceeded_error_on_.2Fproject_filesystems " Disk quota exceeded"){.wikilink} error message. Since `-a` includes `-p` and `-g` by default, the `--no-g --no-p` options should be added, like so

where LOCALNAME can be a directory or file preceded by its path location and somedir will be created if it doesn\'t exist. The `-z` option compresses files (not in the default file suffixes `--skip-compress` list) and requires additional cpu resources while the `-h` option makes transferred file sizes human readable. If you are transferring very large files add the `--partial` option so interrupted transfers maybe restarted:

The `--progress` option will display the percent progress of each file as its transferred. If you are transferring very many smaller files, then it maybe more desirable to display a single progress bar that represents the transfer progress of all files: progress2 LOCALNAME someuser@nibi.alliancecan.ca:projects/def-professor/someuser/somedir/}} The above rsync examples all involve transfers from a local system into a project directory on a remote system. Rsync transfers from a remote system into a project directory on a local system work in much the same way, for example:

where REMOTENAME can be a directory or file preceded by its path location and somedir will be created if it doesn\'t already exist. In its simplest incarnation rsync can also be used locally within a single system to transfer a directory or file (from home or scratch) into project by dropping the cluster name:

where somedir will be created if it doesn\'t already exist before copying LOCALNAME into it. For comparison purposes, the copy command can similarely be used to transfer LOCALNAME from home to project by doing: \"mode,timestamps\" LOCALNAME \~/projects/def-professor/someuser/somedir/}} however unlike rsync, if LOCALNAME is a directory, it will be renamed to somedir if somedir does not exist.

### Using checksums to check if files match {#using_checksums_to_check_if_files_match}

If Globus is unavailable between the two systems being synchronized and Rsync is taking too long, then you can use a [checksum](https://en.wikipedia.org/wiki/Checksum) utility on both systems to determine if the files match. In this example we use `sha1sum`.

`xargs -0 sha1sum  tee checksum-result.log`

}}

This command will create a new file called checksum-result.log in the current directory; the file will contain all of the checksums for the files in /home/username/. It will also print out all of the checksums to the screen as it goes. If you have a lot of files or very large files you may want to run this command in the background, in a [screen](https://en.wikipedia.org/wiki/GNU_Screen) or [tmux](https://en.wikipedia.org/wiki/Tmux) session; anything that allows it to continue if your [SSH](https://docs.alliancecan.ca/SSH "SSH"){.wikilink} connection times out.

After you run it on both systems, you can use the `diff` utility to find files that don\'t match.

It is possible that the `find` command will crawl through the directories in a different order, resulting in a lot of false differences so you may need to run `sort` on both files before running diff such as:

## SFTP

[SFTP](https://en.wikipedia.org/wiki/SSH_File_Transfer_Protocol) (Secure File Transfer Protocol) uses the SSH protocol to transfer files between machines which encrypts data being transferred.

For example, you can connect to a remote machine at `ADDRESS` as user `USERNAME` with SFTP to transfer files like so:

``` console
[name@server]$ sftp USERNAME@ADDRESS
The authenticity of host 'ADDRESS (###.###.###.##)' can't be established.
RSA key fingerprint is ##:##:##:##:##:##:##:##:##:##:##:##:##:##:##:##.
Are you sure you want to continue connecting (yes/no)? yes
Warning: Permanently added 'ADDRESS,###.###.###.##' (RSA) to the list of known hosts.
USERNAME@ADDRESS's password:
Connected to ADDRESS.
sftp>
```

or using an [SSH Key](https://docs.alliancecan.ca/SSH_Keys "SSH Key"){.wikilink} for authentication using the `-i` option

``` console
[name@server]$ sftp -i /home/name/.ssh/id_rsa USERNAME@ADDRESS
Connected to ADDRESS.
sftp>
```

which returns the `sftp>` prompt where commands to transfer files can be issued. To get a list of commands available to use at the sftp prompt enter the `help` command.

There are also a number of graphical programs available for Windows, Linux and Mac OS, such as [WinSCP](https://winscp.net/eng/index.php) and [MobaXterm](http://mobaxterm.mobatek.net/) (Windows), [filezilla](https://filezilla-project.org) (Windows, Mac, and Linux), and [cyberduck](https://cyberduck.io/?l=en) (Mac and Windows).

## SCP

SCP stands for [`<i>`{=html}Secure Copy Protocol`</i>`{=html}](https://en.wikipedia.org/wiki/Secure_copy). Like SFTP it uses the SSH protocol to encrypt data being transferred. It does not support synchronization like [Globus](https://docs.alliancecan.ca/Globus "Globus"){.wikilink} or [rsync](https://docs.alliancecan.ca/Transferring_data#Rsync "rsync"){.wikilink}. Some examples of the most common use of SCP include

which will copy the file `foo.txt` from the current directory on my local computer to the directory `$HOME/work` on the cluster [Rorqual](https://docs.alliancecan.ca/Rorqual/en "Rorqual"){.wikilink}. To copy a file, `output.dat` from my project space on the cluster [Fir](https://docs.alliancecan.ca/Fir "Fir"){.wikilink} to my local computer I can use a command like

Many other examples of the use of SCP are shown [here](http://www.hypexr.org/linux_scp_help.php). Note that you always execute this `scp` command on your local computer, not the remote cluster - the SCP connection, regardless of whether you are transferring data to or from the remote cluster, should always be initiated from your local computer.

SCP supports the option `-r` to recursively transfer a set of directories and files. We `<b>`{=html}recommend against using `scp -r``</b>`{=html} to transfer data into `/project` because the setgid bit is turned off in the created directories, which may lead to `Disk quota exceeded` or similar errors if files are later created there (see [ Disk quota exceeded error on /project filesystems](https://docs.alliancecan.ca/Frequently_Asked_Questions#Disk_quota_exceeded_error_on_.2Fproject_filesystems " Disk quota exceeded error on /project filesystems"){.wikilink}).

`<b>`{=html}`<big>`{=html}\*\*\*Note\*\*\*`</big>`{=html}`</b>`{=html} if you chose a custom SSH key name, `<i>`{=html}i.e.`</i>`{=html} something other than the default names: `id_dsa`, `id_ecdsa`, `id_ed25519` and `id_rsa`, you will need to use the `-i` option of scp and specify the path to your private key before the file paths via

## Prevention and Troubleshooting {#prevention_and_troubleshooting}

### Unable to read data {#unable_to_read_data}

Before initiating any transfer, make sure you can read all the contents of the directories you would like to transfer. On a Linux system, the following command lists all items not readable to you:

### Unable to write new data {#unable_to_write_new_data}

- Double-check the `<b>`{=html}[storage usage](https://docs.alliancecan.ca/Storage_and_file_management#Overview "storage usage"){.wikilink}`</b>`{=html} and make sure enough space and enough files are available.
  - On some clusters, the filesystem automatically compresses your files and reports the space usage by the disk usage of the compressed data. On other clusters, the space usage is reported by the apparent size of your files. Therefore, 1 TB of compressed data on one cluster may become 2 TB of data on the next cluster.
  - Before transferring a dataset, it is possible to get its apparent size with the option `-b` of the `du` command:

<!-- -->

- Double-check the `<b>`{=html}[filesystem permissions](https://docs.alliancecan.ca/Sharing_data "filesystem permissions"){.wikilink}`</b>`{=html} and make sure you have the write permission at the location where you are trying to transfer new files.
