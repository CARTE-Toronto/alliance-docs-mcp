---
title: "Windows Subsystem for Linux (WSL)/en"
url: "https://docs.alliancecan.ca/wiki/Windows_Subsystem_for_Linux_(WSL)/en"
category: "General"
last_modified: "2024-04-02T21:23:48Z"
page_id: 25182
display_title: "Windows Subsystem for Linux (WSL)"
---

`<languages />`{=html}

# Introduction

Windows Subsystem for Linux (WSL) is a feature of the Windows operating system that allows you to run a Linux environment on your Windows machine, without requiring a full-featured Virtual Machine application or other complex method such as dual-booting. Using WSL allows you to have access to both Windows and Linux applications and files at the same time in a seamless, integrated manner.

This setup is of particular interest if you are running a Windows-based computer and require access to Linux-based Alliance resources. It allows you to use Linux-based tools to connect and transfer data to and from Alliance resources, while having access to your familiar Windows environment at the same time.

This article is a quick introduction to basic tasks that WSL can assist with. If more detailed documentation is required, please refer to the [documentation provided by Microsoft about WSL](https://learn.microsoft.com/en-us/windows/wsl/).

# Installing Windows Subsystem for Linux {#installing_windows_subsystem_for_linux}

The installation and setup of WSL is [covered in detail by Microsoft.](https://learn.microsoft.com/en-us/windows/wsl/install)

To get started quickly on a Windows 10/11 machine that has not yet had WSL installed, the following steps will install WSL and Ubuntu (a popular version of Linux):

1.  Save your work, as this process requires a reboot.
2.  Click on the start button and begin typing `<i>`{=html}command prompt`</i>`{=html}.
3.  Right-click on the Command Prompt application and select `<i>`{=html}Run as administrator`</i>`{=html}. Accept any security prompt that appears.
4.  In the Command Prompt window, type the following command, and wait for it to complete:
5.  Restart your machine.

# Launching Ubuntu for the first time {#launching_ubuntu_for_the_first_time}

When your computer completes its reboot, you will have a new application available in the Start menu: `<i>`{=html}Ubuntu`</i>`{=html}. Upon launching this application for the first time, WSL will decompress some files and prepare the Ubuntu Linux environment for use. When complete, you will be asked to configure your Linux user and set a password.

Take note:

- This user name is `<b>`{=html}unique to the Linux system only`</b>`{=html}, and does not need to match the Windows username.
- If you later install multiple different Linux environments within WSL, each one of them will have its own users and passwords (they are not shared).

1.  At the prompt `<i>`{=html}Enter new UNIX username`</i>`{=html}, enter your desired username and press enter.
2.  At the prompt `<i>`{=html}Enter new UNIX password`</i>`{=html}, enter your desired password and press enter. You will not see characters as you type them; this is normal.

Your WSL/Ubuntu Linux setup is complete and you can now use it.

# File access between Windows and Linux {#file_access_between_windows_and_linux}

Linux environments operating in WSL are essentially equivalent to virtual machines. As such, they do not inherently share all of the same access to data stored within each environment; however, WSL has gone to great lengths to bridge this gap by two means:

1.  By automatically mounting (attaching) your Windows drives within the Linux folder structure at `<i>`{=html}/mnt/`</i>`{=html}.
2.  By adding a Linux entry in the Windows Explorer sidebar that provides direct access to files stored within Linux.

These integrations allow you to transfer data easily between the two systems. As an example, the common Windows drive `<i>`{=html}C:\\`</i>`{=html} would be available in Linux at `<i>`{=html}/mnt/c`</i>`{=html}, and the Linux user's home folder would be available in Windows Explorer at `<i>`{=html}Linux \> Ubuntu \> home \> username`</i>`{=html}.

There are some notable differences between the way that Windows and Linux handle file paths:

- Windows uses the backslash character (\\) between directories, whereas Linux uses a forward slash (/).
- Linux uses a case-sensitive approach to file and directory names, meaning that uppercase and lowercase letters are different: FILE.TXT, file.txt, and FILE.txt are all different files in Linux. Windows is case-insensitive, so all three of the examples given prior would point to the same file in Windows.

## Accessing Windows files from Linux (command line) {#accessing_windows_files_from_linux_command_line}

1.  Find the full path of the file or folder on Windows.
2.  Note the drive letter (e.g., `<i>`{=html}C:\\`</i>`{=html}).
3.  Replace the drive letter with `<i>`{=html}/mnt/{letter}/`</i>`{=html}.
4.  Transform all of the backslashes to forward slashes.

Examples:

- `<i>`{=html}C:\\Users\\user1\\Documents\\File1.txt`</i>`{=html} is located at `<i>`{=html}/mnt/c/Users/user1/Documents/File1.txt`</i>`{=html} in Linux.
- `<i>`{=html}D:\\Data\\Project\\Dataset\\`</i>`{=html} is located at `<i>`{=html}/mnt/d/Data/Project/Dataset/`</i>`{=html} in Linux.

## Accessing Linux files from Windows (2 methods) {#accessing_linux_files_from_windows_2_methods}

### Method 1 {#method_1}

1.  Find the full path of the file or folder on Linux.
2.  Use Windows Explorer's sidebar to find the Linux entry (usually near the bottom) and expand it.
3.  Select the Linux environment that contains the file (`<i>`{=html}Ubuntu`</i>`{=html} by default).
4.  Navigate through the same folder structure from step 1 to find the file/folder.

Example:

- `<i>`{=html}/home/username/file1.txt`</i>`{=html} is located at `<i>`{=html}Linux \> Ubuntu \> home \> username \> file1.txt`</i>`{=html} in Windows Explorer.

### Method 2 {#method_2}

1.  Open a WSL command line and change directory to where the file is stored.
2.  Run to open a Windows Explorer window at the intended directory (the trailing period is important, and directs Explorer to open the current directory).

# Transferring data using WSL {#transferring_data_using_wsl}

A common use case of WSL is to use it for transferring data to Alliance resources using programs such as [FileZilla](https://filezilla-project.org/). Often, support for [multifactor authentication](https://docs.alliancecan.ca/Multifactor_authentication "multifactor authentication"){.wikilink} is stronger inside Linux (and by extension WSL) due to various technical factors. You can easily install such programs inside the Ubuntu WSL environment; in the case of FileZilla:

The application is now installed and you can launch it from either from the Linux command line with `filezilla` or by the Windows start menu.

When you are browsing the filesystem of Linux using such tools, remember that your Windows files can be found under `<i>`{=html}/mnt/{drive letter}`</i>`{=html} by default, and you can access them directly `<b>`{=html}without`</b>`{=html} needing to first copy them into the Linux environment.

For more information about transferring data to Alliance resources, please refer to the [Transferring data](https://docs.alliancecan.ca/Transferring_data "Transferring data"){.wikilink} page.
