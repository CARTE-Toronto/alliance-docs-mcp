---
title: "Prolonging terminal sessions/en"
url: "https://docs.alliancecan.ca/wiki/Prolonging_terminal_sessions/en"
category: "General"
last_modified: "2023-07-04T16:40:30Z"
page_id: 22803
display_title: "Prolonging terminal sessions"
---

`<languages />`{=html} To work on our clusters, most users will need to use [SSH](https://docs.alliancecan.ca/SSH "SSH"){.wikilink} to connect to the cluster for job submission and monitoring, editing files and so forth. Keeping this SSH connection alive for a long period of time, ranging from hours to days, may be necessary for some users and this page proposes a variety of techniques for keeping such a terminal session alive.

# SSH configuration {#ssh_configuration}

One simple solution is to modify the configuration of your SSH client to prolong the connection. On MacOS and Linux the client configuration is found in `$HOME/.ssh/config` while in Windows it is located in `C:\Users\``<username>`{=html}`\.ssh\config`. Note that the file may not exist initially, so you will need to create it; you should add the lines

    Host *
        ServerAliveInterval 240

This addition will ensure the transmission of a sign-of-life signal over the SSH connection to any remote server (such as an Alliance cluster) every 240 seconds, i.e. four minutes, which should help to keep your SSH connection alive even if it is idle for several hours.

# Terminal multiplexers {#terminal_multiplexers}

The programs `tmux` and `screen` are examples of a terminal multiplexer---a program which allows you to detach your terminal session entirely, where it will keep on running on its own until you choose to reattach to it. With such a program, you can log out from the cluster, turn off the workstation or hibernate the laptop you use to connect to the cluster and when you\'re ready to start working again the next day, reattach to your session and start from right where you left off.

## tmux

The [tmux](https://en.wikipedia.org/wiki/Tmux) software is a terminal multiplexer, allowing multiple virtual sessions in a single terminal session. You can thus disconnect from an SSH session without interrupting its process(es).

Here are some introductions to tmux:

- [\"The Tao of tmux\"](https://leanpub.com/the-tao-of-tmux/read), an online book
- [\"Getting Started With TMUX\"](https://www.youtube.com/watch?v=252K9lrRdMU), a 24-minute video
- [\"Turbo boost your interactive experience on the cluster with tmux\"](https://www.youtube.com/watch?v=Y1Of3S5iVog), a 58-minute video

### Cheat sheet {#cheat_sheet}

For a complete reference, see [this page](http://hyperpolyglot.org/multiplexers).

  Command      Description
  ------------ -------------------------------------------------------------------------------
  `tmux`       Start a server
  `Ctrl+B D`   Disconnect from server
  `tmux a`     Reconnect to server
  `Ctrl+B C`   Create a new window
  `Ctrl+B N`   Go to next window
  `Ctrl+B [`   Enable \"copy\" mode, allowing to scroll with the mouse and Page-Up Page-Down
  `Esc`        Disable \"copy\" mode

### Launch tmux inside a job submitted through tmux {#launch_tmux_inside_a_job_submitted_through_tmux}

If you submit a job with tmux and try to start tmux within the same job, you will get the `lost server` error message. This happens because the `$TMUX` environment variable pointing to the tmux server is propagated to the job. The value of the variable is not valid and you can reset it with:

However, nested use of tmux is not recommended. To send commands to a nested tmux, one has to hit `Ctrl+B` twice; for example, to create a new window, one has to use `Ctrl+B Ctrl+B C`. Consider using [ GNU Screen](https://docs.alliancecan.ca/#GNU_Screen " GNU Screen"){.wikilink} inside your job (if you are using tmux on a login node).

## GNU Screen {#gnu_screen}

The [GNU Screen](https://en.wikipedia.org/wiki/GNU_Screen) program is another widely used terminal multiplexer. To create a detached terminal session, you can use the following command

It\'s a good idea to give a descriptive name to your terminal sessions, making it easier to identify them later. You can use the command `screen -list` to see a list of your detached terminal sessions on this node,

You can attach to one of your sessions using the command `screen -d -r ``<session name>`{=html}.
