---
title: "Nextcloud/en"
url: "https://docs.alliancecan.ca/wiki/Nextcloud/en"
category: "General"
last_modified: "2024-06-12T15:51:09Z"
page_id: 14336
display_title: "Nextcloud"
---

We provide Nextcloud, a Dropbox-like cloud storage service, for all Alliance users. You can use your Alliance username and password to log in to the Nextcloud server. A complete Nextcloud user manual is available from the official Nextcloud documentation. A manual is also available as a PDF document under your account once you connect. All data transfers between local devices and Alliance's Nextcloud are encrypted.

The Nextcloud service is aimed at users with relatively small datasets (up to 100 GB). For anything larger, we recommend using the Globus service.

If you are not familiar with the concept of Nextcloud, you may try the demo on the Nextcloud website.

We recommend taking this opportunity to take a look at your data and do some cleanup: remove data you no longer need, check with whom you share your data, etc.

==Alliance Nextcloud service description==

* Server URL: https://nextcloud.computecanada.ca
* Server Location: Simon Fraser University, Burnaby, BC
* Fixed Quota: 100 GB per user
* Backup Policy: Daily backup without offsite copy
* Access Methods: Web interface, Nextcloud Desktop Sync Client, Nextcloud mobile apps, and any WebDAV client
* Documentation: PDF and online

==Using the Nextcloud web interface==

To use the web interface, log in to Alliance Nextcloud from a web browser using your Alliance username and password. You can upload and download files between Nextcloud and your mobile device or computer, edit files, and share files with other Alliance users. For more information, see the Nextcloud user manual.

==Using Nextcloud Desktop Synchronization Client and mobile apps==

You can download the Nextcloud Desktop Sync Client or Nextcloud mobile apps to synchronize data from your computer or your mobile device respectively. Once installed, the software will "sync" everything between your Nextcloud folder and your local folder. It may take some time to synchronize all data. You can make changes to files locally and they will be updated in Nextcloud automatically.

==Using WebDAV clients==

In general, you can use any WebDAV clients to "mount" a Nextcloud folder to your computer using the following WebDAV URL: https://nextcloud.computecanada.ca/remote.php/webdav/

Once mounted, you can drag and drop files between the WebDAV drive and your local computer.

Mac OSX: Select Go -> Connect to the Server, enter the WebDAV URL for the Server Address, and click Connect. You will be asked for your username and password to log in. After authentication, you will see a WebDAV drive on your Mac.

Windows: Use the "Map Network Drive ..." option, select a drive letter, then use WebDAV URL https://nextcloud.computecanada.ca/remote.php/webdav/ in the Folder field.

You may also consider using Cyberduck or other clients instead. Cyberduck is available for OSX and Windows.

Linux: There are many WebDAV applications available for Linux. Consult the Nextcloud user manual for recommendations.

===Detail: WebDAV vs Synchronization Client===

The WebDAV clients mount your Nextcloud storage on your computer.  Files are not copied; for example, when you edit a file, you edit the original file on the Alliance Nextcloud system at Simon Fraser University.

When you connect with a Synchronization client, the first thing the client does is synchronize your files stored in the Alliance Nextcloud system with a copy of those files on your own computer. All files that are different get downloaded to your own client. When files are changed, they are re-copied to all the synchronized systems to ensure that the files are the same everywhere.  The synchronization copies can take a lot of time when you (and/or your collaborators) change files frequently. The advantage is that you can work on the files offline, i.e., when you do not have network connectivity. They will be synchronized when network connectivity is re-established.

==Using UNIX command line tools==

You can use any available WebDAV command line clients, like curl and cadaver, to copy files between your Unix computer and Nextcloud. Command line tools are useful when you want to copy data between a remote server you log in to and Nextcloud.

curl is usually installed on Mac OSX and Linux systems and can be used to upload and download files using a URL.

===Upload a file using curl===

===Download a file using curl===

===Upload and download files using rclone===

Unlike curl, rclone lets you create a configuration once for each remote device and use it repeatedly without having to enter the service details and your password every time. The password will be stored encrypted in ~/.config/rclone/rclone.conf on the computer or server where the rclone command is used.

First, install rclone on your computer if it has a Unix-like environment.

If used from our clusters, please note that it is no necessary to install rclone as it is already available:

$ [name@server ~] $ which rclone
$ /cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/bin/rclone

Next, configure a remote storage device profile with

$ rclone config

You now have the option to edit an existing remote device, create a new remote device, delete a remote device, and so on. Let's say we want to create a new remote serice profile called nextcloud:

choose "n"  for "New remote"
Enter name for new remote --> nextcloud
Type of storage to configure --> 52 / WebDAV
URL of http host to connect to --> https://nextcloud.computecanada.ca/remote.php/dav/files/
Name of the WebDAV site/service/software you are using --> 2 / Nextcloud
User name -->
choose "y" for "Option pass"
Password -->
Leave "Option bearer_token" empty
choose "no" for "Edit advanced config"
choose "yes" for "Keep this 'nextcloud' remote"
choose "q" to quit config

You should now be able to see your new remote service profile in the list of configured ones with

 $ rclone listremotes

You can probe available disk space with

 $ rclone about nextcloud:

To upload a file, run

 $ rclone copy /path/to/local/file nextcloud:remote/path

To download a file, run

 $ rclone copy nextcloud:remote/path/file .

==Sharing files using Nextcloud==

When you select a file or directory to share, type the user’s first name, last name, or username and the list of matched users registered in CCDB will be displayed in “Firstname Lastname (username)” format. Please review the name carefully as some are very similar; in doubt, enter the username which is unique. You can also share files with a group using their CCDB group name (default, RPP, RRG, or other shared groups).
To share a file with people who don’t have an Alliance account, use the Share link option and  provide their email address. Nextcloud will send an email notification with a link to access the file.