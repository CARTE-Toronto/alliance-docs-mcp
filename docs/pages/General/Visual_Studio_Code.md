---
title: "Visual Studio Code/en"
url: "https://docs.alliancecan.ca/wiki/Visual_Studio_Code/en"
category: "General"
last_modified: "2025-10-21T20:16:42Z"
page_id: 24790
display_title: "Visual Studio Code"
---

Visual Studio Code is an integrated development environment (IDE) from Microsoft which can be used for local development with numerous extensions and is highly customizable.

* Use VS Code locally and avoid connecting it to the systems. Save your changes to your project files with Git, and pull the changes onto the systems when ready to test.
* Use nano or vim to edit files directly on the systems.
* For debugging and quick testing, you can load the code-server module.
* When all the above are not possible, one can configure VS Code for remote connections.

__FORCETOC__

= Local usage =
The advantages of using VS Code locally are
* speed & stability: running VS Code locally means fewer network interruptions and faster performance, which is ideal for iterative development;
* direct access: you can interact with files, extensions, and terminals directly on your machine with zero latency;
* offline capability: you’re not tied to an internet connection or remote server, so you can code anytime, anywhere.

It is recommended to develop locally with VS Code. You are then able to customize and extend VS Code with your preferred extensions and language.

Once you are ready to test your project on the systems, you can save your changes into a Git repository, push them to a remote host like GitHub or GitLab, then connect to the system and pull your changes to perform the test.

To learn more on how to work with source control, please see VS Code Source Control.

Once you have saved and pushed your changes to your remote repository, connect to the system via the terminal.

Then clone your repository (if it does not exist).

or change directory to your repository and pull the changes with

You can then test your changes in a short (with minimal resources) interactive job.

= File edition on the systems =
While VS Code is great for local development, sometimes you need direct access to files on a remote system. In such cases, terminal-based editors like nano or vim offer a lightweight and efficient way to edit files directly from the command line.

If you prefer a graphical interface, the JupyterLab text editor provides a versatile alternative. It supports Markdown, Python scripts, and other formats.

= Debugging and testing =
If you need to debug or test your code on the systems, you can start a code-server instance from Jupyter Lab.

# Access one of the options to launch JupyterLab.
# Select minimal resources and start an interactive JupyterLab job.
# On the Launcher tab, click on the VS Code launcher button.

The code-server module has several common extensions already available, but we can add more upon request.

== Custom extension installation ==
TBD...

= Configuration of VS Code for remote connection =
If none of the above works for your case, one can configure VS Code to connect to a remote host with the Remote SSH extension.

== SSH configuration ==
If not done already, generate your SSH key and add your public SSH key on the CCDB.

Then create (or add) an SSH configuration file to your local computer:

== Local configuration ==
1. In VS Code, open the Command Palette: Press Ctrl+Shift+P (Windows/Linux) or Cmd+Shift+P (macOS).

2. Open the user settings (Preferences: Open User Settings (JSON)) and paste (or merge) the following configuration:

3. Save it and restart VS Code.

== Remote system configuration ==
1. Log in to the system via an external terminal.

2. Create the directory.

3. Create the settings.json machine configuration.

4. Copy the configuration below. You may need to manually merge settings with your own if any already.

== Connecting ==
# Open the Command Palette in VS Code: Press Ctrl+Shift+P (Windows/Linux) or Cmd+Shift+P (macOS).
# Type remote and then select Connect to Host...
# Choose the host (remote system) and confirm.

You'll now be connected to a login node.

== Closing your connection ==
# Open the Command Palette in VS Code: Press Ctrl+Shift+P (Windows/Linux) or Cmd+Shift+P (macOS).
# Type remote and then select Remote-SSH: Kill VS Code Server on Host...
# Choose the host (remote system) and confirm.
# Open the File menu, and select Close Remote Connection.

== Advanced - Connecting to an interactive compute node ==
The following is only needed for advanced usage.

Update your ssh configuration to add the following lines:

# In an external terminal, connected to the system via an ssh connection, start a new interactive job (with salloc) with at least 2000M of memory.
## Note the allocated compute node name.
## If you need to work with SLURM_* environment variables in VS Code, save them all in a source file:  grep SLURM_  sed -e 's/^\(.*\)\(.*\)$/export \1"\2"/g' > slurm_var.sh}}
# In VS Code, start a new remote session with the name of the allocated compute node.
## Press F1 or Ctrl+Shift+P to start the command prompt > in the Command Palette.
## Start typing Remote and select Remote-SSH: Connect to Host... > Remote-SSH: Connect to Host...
## Enter the noted compute node name.
### If you get prompted for the type of operating system, select Linux.
# If you need to work with SLURM_* environment variables, navigate to the working directory in a VS Code terminal and source the slurm_var.sh file.

= Special notes =
* VScode is banned on tamIA login nodes.