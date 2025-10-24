---
title: "Visual Studio Code/en"
url: "https://docs.alliancecan.ca/wiki/Visual_Studio_Code/en"
category: "General"
last_modified: "2025-10-21T20:16:42Z"
page_id: 24790
display_title: "Visual Studio Code"
---

`<languages />`{=html}

[Visual Studio Code](https://code.visualstudio.com/) is an integrated development environment (IDE) from [Microsoft](https://www.microsoft.com) which can be used for local development with numerous extensions and is highly customizable.

-   Use VS Code locally and avoid connecting it to the systems. Save your changes to your project files with Git, and pull the changes onto the systems when ready to test.
-   Use nano or vim to edit files directly on the systems.
-   For debugging and quick testing, you can load the `code-server` module.
-   When all the above are not possible, one can configure VS Code for remote connections.

\_\_FORCETOC\_\_

# Local usage {#local_usage}

The advantages of using VS Code locally are

-   speed & stability: running VS Code locally means fewer network interruptions and faster performance, which is ideal for iterative development;
-   direct access: you can interact with files, extensions, and terminals directly on your machine with zero latency;
-   offline capability: you're not tied to an internet connection or remote server, so you can code anytime, anywhere.

It is recommended to develop locally with VS Code. You are then able to customize and extend VS Code with your preferred extensions and language.

Once you are ready to test your project on the systems, you can save your changes into a Git repository, push them to a remote host like GitHub or GitLab, then connect to the system and pull your changes to perform the test.

To learn more on how to work with source control, please see [VS Code Source Control](https://code.visualstudio.com/docs/sourcecontrol/overview).

Once you have saved and pushed your changes to your remote repository, connect to the system via the terminal.

Then clone your repository (if it does not exist).

or change directory to your repository and pull the changes with

You can then test your changes in a short (with minimal resources) [interactive job](https://docs.alliancecan.ca/Running_jobs#Interactive_jobs "wikilink").

# File edition on the systems {#file_edition_on_the_systems}

While VS Code is great for local development, sometimes you need direct access to files on a remote system. In such cases, terminal-based editors like `nano` or `vim` offer a lightweight and efficient way to edit files directly from the command line.

If you prefer a graphical interface, the [JupyterLab](https://docs.alliancecan.ca/wiki/JupyterLab) text editor provides a versatile alternative. It supports Markdown, Python scripts, and other formats.

# Debugging and testing {#debugging_and_testing}

If you need to debug or test your code on the systems, you can start a [`code-server` instance from Jupyter Lab](https://docs.alliancecan.ca/JupyterLab#VS_Code "wikilink").

1.  Access one of the [options to launch JupyterLab](https://docs.alliancecan.ca/JupyterLab#Launching_JupyterLab "wikilink").
2.  Select minimal resources and start an interactive JupyterLab job.
3.  On the Launcher tab, click on the VS Code launcher button.

The `code-server` module has several common extensions already available, but we can add more upon request.

## Custom extension installation {#custom_extension_installation}

TBD\...

# Configuration of VS Code for remote connection {#configuration_of_vs_code_for_remote_connection}

If none of the above works for your case, one can configure VS Code to connect to a remote host with the Remote SSH extension.

## SSH configuration {#ssh_configuration}

If not done already, [generate your SSH key](https://docs.alliancecan.ca/SSH_Keys#Generating_an_SSH_Key "wikilink") and [add your `<i>`{=html}public`</i>`{=html} SSH key on the CCDB](https://docs.alliancecan.ca/SSH_Keys#Installing_your_key "wikilink").

Then create (or add) an SSH configuration file to your local computer:

## Local configuration {#local_configuration}

1\. In VS Code, open the Command Palette: Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (macOS).

2\. Open the user settings (`Preferences: Open User Settings (JSON)`) and paste (or merge) the following configuration: `{{File
  |name=local-settings.json
  |contents=
{
  // file-watch + search
  "files.watcherExclude": {
    "**/.git/**": true,
    "**/node_modules/**": true,
    "**/dist/**": true,
    "**/build/**": true,
  },
  "search.exclude": {
    "**/.git/objects/**": true,
    "**/.git/subtree-cache/**": true,
    "**/node_modules/**": true,
  },
  "search.maxThreads": 2,
  "search.ripgrep.maxThreads": 2,
  "search.useIgnoreFiles": true,

  // extensions & updates
  "remote.extensionKind": {
    "*": [
      "ui"
    ],
    "ms-python.python": [
      "ui"
    ]
  },
  "remote.defaultExtensionsIfInstalledLocally": [
    "GitHub.vscode-pull-request-github"
  ],

  // remote-ssh
  "remote.SSH.showLoginTerminal": false,
  "remote.SSH.enableDynamicForwarding": false,
  "remote.SSH.enableServerAutoShutdown": 30,

  "workbench.startupEditor": "none",
}
}}`{=mediawiki}

3\. Save it and restart VS Code.

## Remote system configuration {#remote_system_configuration}

1\. Log in to the system via an external terminal.

2\. Create the directory.

3\. Create the `settings.json` machine configuration.

4\. Copy the configuration below. You may need to manually merge settings with your own if any already. `{{File
  |name=system-settings.json
  |contents=
{
  // file-watch + search
  "files.watcherExclude": {
    "**/.git/**": true,
    "**/node_modules/**": true,
    "**/dist/**": true,
    "**/build/**": true,
    "/**": true,
  },
  "search.exclude": {
    "**/.git/objects/**": true,
    "**/.git/subtree-cache/**": true,
    "**/node_modules/**": true,
    "/**": true,
  },
  "search.followSymlinks": false,
  "search.maxThreads": 2,
  "search.ripgrep.maxThreads": 2,
  "search.useIgnoreFiles": true,
  "search.searchOnType": false,

  // extensions & updates
  "extensions.autoCheckUpdates": false,
  "extensions.autoUpdate": false,
  "update.mode": "none",
  "remote.extensionKind": {
    "*": [
      "ui"
    ],
    "ms-python.python": [
      "ui"
    ]
  },

  // Copilot
  "chat.agent.enabled": false,
  "github.copilot.enable": {
    "*": false,
  },
  "remote.defaultExtensionsIfInstalledLocally": [
    "GitHub.vscode-pull-request-github"
  ],

  // telemetry & git
  "telemetry.enableTelemetry": false,
  "telemetry.enableCrashReporter": false,
  "telemetry.telemetryLevel": "off",
  "telemetry.feedback.enabled": false,
  "git.autofetch": false,
  "git.enableStatusBarSync": false,

  // remote-ssh
  "remote.SSH.showLoginTerminal": false,
  "remote.SSH.enableDynamicForwarding": false,
  "remote.SSH.enableServerAutoShutdown": 30,

  "workbench.startupEditor": "none",
}
}}`{=mediawiki}

## Connecting

1.  Open the Command Palette in VS Code: Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (macOS).
2.  Type `remote` and then select `Connect to Host...`
3.  Choose the host (remote system) and confirm.

You\'ll now be connected to a **login node**.

## Closing your connection {#closing_your_connection}

1.  Open the Command Palette in VS Code: Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (macOS).
2.  Type `remote` and then select `Remote-SSH: Kill VS Code Server on Host...`
3.  Choose the host (remote system) and confirm.
4.  Open the File menu, and select `Close Remote Connection`.

## Advanced - Connecting to an interactive compute node {#advanced___connecting_to_an_interactive_compute_node}

The following is only needed for advanced usage.

Update your ssh configuration to add the following lines: `<tabs>`{=html} `<tab name="Narval">`{=html}

`</tab>`{=html} `<tab name="Rorqual">`{=html}

`</tab>`{=html} `</tabs>`{=html}

1.  In an external terminal, connected to the system via an ssh connection, start a new `<b>`{=html}[interactive job](https://docs.alliancecan.ca/Running_jobs#Interactive_jobs "wikilink")`</b>`{=html} (with `salloc`) with at least 2000M of memory.
    1.  Note the allocated compute node name.
    2.  If you need to work with `SLURM_*` environment variables in VS Code, save them all in a *source* file: grep SLURM\_ sed -e \'s/\^$.*$$.*$\$/export \\1\"\\2\"/g\' \> slurm_var.sh}}
2.  In VS Code, start a new remote session with the name of the allocated compute node.
    1.  Press `F1` or `Ctrl+Shift+P` to start the command prompt `>` in the Command Palette.
    2.  Start typing `<i>`{=html}Remote`</i>`{=html} and select `<i>`{=html}Remote-SSH: Connect to Host\... `<b>`{=html}\> Remote-SSH: Connect to Host\...`</i>`{=html}`</b>`{=html}
    3.  Enter the noted compute node name.
        1.  If you get prompted for the type of operating system, select `<b>`{=html}Linux`</b>`{=html}.
3.  If you need to work with `SLURM_*` environment variables, navigate to the working directory in a VS Code terminal and *source* the `slurm_var.sh` file.

# Special notes {#special_notes}

-   VScode is banned on tamIA login nodes.
