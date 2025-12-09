---
title: "Windows Subsystem for Linux (WSL)/en"
url: "https://docs.alliancecan.ca/wiki/Windows_Subsystem_for_Linux_(WSL)/en"
category: "General"
last_modified: "2024-04-02T21:23:48Z"
page_id: 25182
display_title: "Windows Subsystem for Linux (WSL)"
---

= Introduction =

Windows Subsystem for Linux (WSL) is a feature of the Windows operating system that allows you to run a Linux environment on your Windows machine, without requiring a full-featured Virtual Machine application or other complex method such as dual-booting. Using WSL allows you to have access to both Windows and Linux applications and files at the same time in a seamless, integrated manner.

This setup is of particular interest if you are running a Windows-based computer and require access to Linux-based Alliance resources. It allows you to use Linux-based tools to connect and transfer data to and from Alliance resources, while having access to your familiar Windows environment at the same time.

This article is a quick introduction to basic tasks that WSL can assist with. If more detailed documentation is required, please refer to the documentation provided by Microsoft about WSL.

= Installing Windows Subsystem for Linux =

The installation and setup of WSL is covered in detail by Microsoft.

To get started quickly on a Windows 10/11 machine that has not yet had WSL installed, the following steps will install WSL and Ubuntu (a popular version of Linux):
# Save your work, as this process requires a reboot.
# Click on the start button and begin typing command prompt.
# Right-click on the Command Prompt application and select Run as administrator. Accept any security prompt that appears.
# In the Command Prompt window, type the following command, and wait for it to complete:
# Restart your machine.

= Launching Ubuntu for the first time =

When your computer completes its reboot, you will have a new application available in the Start menu: Ubuntu. Upon launching this application for the first time, WSL will decompress some files and prepare the Ubuntu Linux environment for use. When complete, you will be asked to configure your Linux user and set a password.

Take note:
* This user name is unique to the Linux system only, and does not need to match the Windows username.
* If you later install multiple different Linux environments within WSL, each one of them will have its own users and passwords (they are not shared).

# At the prompt Enter new UNIX username, enter your desired username and press enter.
# At the prompt Enter new UNIX password, enter your desired password and press enter. You will not see characters as you type them; this is normal.

Your WSL/Ubuntu Linux setup is complete and you can now use it.

= File access between Windows and Linux =
Linux environments operating in WSL are essentially equivalent to virtual machines. As such, they do not inherently share all of the same access to data stored within each environment; however, WSL has gone to great lengths to bridge this gap by two means:
# By automatically mounting (attaching) your Windows drives within the Linux folder structure at /mnt/.
# By adding a Linux entry in the Windows Explorer sidebar that provides direct access to files stored within Linux.

These integrations allow you to transfer data easily between the two systems. As an example, the common Windows drive C:\ would be available in Linux at /mnt/c, and the Linux user’s home folder would be available in Windows Explorer at Linux > Ubuntu > home > username.

There are some notable differences between the way that Windows and Linux handle file paths:

* Windows uses the backslash character (\) between directories, whereas Linux uses a forward slash (/).
* Linux uses a case-sensitive approach to file and directory names, meaning that uppercase and lowercase letters are different: FILE.TXT, file.txt, and FILE.txt are all different files in Linux. Windows is case-insensitive, so all three of the examples given prior would point to the same file in Windows.

== Accessing Windows files from Linux (command line) ==

# Find the full path of the file or folder on Windows.
# Note the drive letter (e.g., C:\).
# Replace the drive letter with /mnt/{letter}/.
# Transform all of the backslashes to forward slashes.

Examples:

* C:\Users\user1\Documents\File1.txt is located at /mnt/c/Users/user1/Documents/File1.txt in Linux.
* D:\Data\Project\Dataset\ is located at /mnt/d/Data/Project/Dataset/ in Linux.

== Accessing Linux files from Windows (2 methods) ==

=== Method 1 ===

# Find the full path of the file or folder on Linux.
# Use Windows Explorer’s sidebar to find the Linux entry (usually near the bottom) and expand it.
# Select the Linux environment that contains the file (Ubuntu by default).
# Navigate through the same folder structure from step 1 to find the file/folder.

Example:

* /home/username/file1.txt is located at Linux > Ubuntu > home > username > file1.txt in Windows Explorer.

=== Method 2 ===

# Open a WSL command line and change directory to where the file is stored.
# Run  to open a Windows Explorer window at the intended directory (the trailing period is important, and directs Explorer to open the current directory).

= Transferring data using WSL =

A common use case of WSL is to use it for transferring data to Alliance resources using programs such as FileZilla. Often, support for multifactor authentication is stronger inside Linux (and by extension WSL) due to various technical factors. You can easily install such programs inside the Ubuntu WSL environment; in the case of FileZilla:

The application is now installed and you can launch it from either from the Linux command line with filezilla or by the Windows start menu.

When you are browsing the filesystem of Linux using such tools, remember that your Windows files can be found under /mnt/{drive letter} by default, and you can access them directly without needing to first copy them into the Linux environment.

For more information about transferring data to Alliance resources, please refer to the Transferring data page.