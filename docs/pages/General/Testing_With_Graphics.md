---
title: "Testing With Graphics"
url: "https://docs.alliancecan.ca/wiki/Testing_With_Graphics"
category: "General"
last_modified: "2019-01-29T18:18:41Z"
page_id: 9029
display_title: "Testing With Graphics"
---

If you need to use graphics while testing your code, e.g. when using a debugger such as DDT or DDD, you have the following options:

# Use the debugjob command {#use_the_debugjob_command}

- You can use the `debugjob` command which automatically provides X-forwarding support.
  ``` bash
  $ ssh  niagara.scinet.utoronto.ca -X

  <!--T:160-->
  USER@nia-login07:~$ debugjob
  debugjob: Requesting 1 nodes for 60 minutes
  xalloc: Granted job allocation 189857
  xalloc: Waiting for resource configuration
  xalloc: Nodes nia0030 are ready for job

  <!--T:161-->
  [USER@nia1265 ~]$
  ```

  # Use the regular queue {#use_the_regular_queue}
- If `debugjob` is not suitable for your case due to the limitations either on time or resources (see above [#Testing](https://docs.alliancecan.ca/#Testing "#Testing"){.wikilink}), then you have to follow these steps:
  You will need two terminals in order to achieve this:

  1.  In the 1st terminal
      - ssh to `niagara.scinet.utoronto.ca` and issue your `salloc` command
      - wait until your resources are allocated and you are assigned the nodes
      - take note of the node where you are logged to, ie. the head node, let\'s say `niaWXYZ`

      ``` bash
      $ ssh  niagara.scinet.utoronto.ca
      USER@nia-login07:~$ salloc --nodes 5 --time=2:00:00

      <!--T:164-->
      .salloc: Granted job allocation 141862
      .salloc: Waiting for resource configuration
      .salloc: Nodes nia1265 are ready for job

      <!--T:165-->
      [USER@nia1265 ~]$
      ```
  2.  On the second terminal:
      - ssh into `niagara.scinet.utoronto.ca` now using the `-X` flag in the ssh command
      - after that `ssh -X niaWXYZ`, ie. you will ssh carrying on the \'-X\' flag into the head node of the job
      - in the `niaWXYZ` you should be able to use graphics and should be redirected by x-forwarding to your local terminal

      ``` bash
      ssh niagara.scinet.utoronto.ca -X
      USER@nia-login07:~$ ssh -X nia1265
      [USER@nia1265 ~]$ xclock   ## just an example to test the graphics, a clock should pop up, close it to exit
      [USER@nia1265 ~]$ module load ddt  ## load corresponding modules, eg. for DDT
      [USER@nia1265 ~]$ ddt  ## launch DDT, the GUI should appear in your screen
      ```

Observations:

- If you are using ssh from a Windows machine, you need to have an X-server, a good option is to use MobaXterm, that already brings an X-server included.
- If you are in Mac OS, substitute -X by -Y
- Instead of using two terminals, you could just use `screen` to request the resources and then detach the session and ssh into the head node directly.
