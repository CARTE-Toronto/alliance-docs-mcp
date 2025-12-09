---
title: "Diskusage Explorer/en"
url: "https://docs.alliancecan.ca/wiki/Diskusage_Explorer/en"
category: "General"
last_modified: "2025-09-10T18:37:53Z"
page_id: 16270
display_title: "Diskusage Explorer"
---

==Content of folders==

Warning: This tool is currently only available on Rorqual and Narval.

You can get a breakdown by folder of how the disk space is being consumed in your /home, /scratch and /project spaces. That information is currently updated once a day and is stored in an SQLite format for fast access.

Here is how to explore your disk consumption, using the example of /project space def-professor as the particular directory to investigate.

=== ncurse user interface ===
Choose a /project space you have access to and want to analyze; for the purpose of this discussion we will use def-professor.

This command loads a browser that shows the resources consumed by all files under any directory tree.

Type c to toggle between consumed disk space and the number of files, q or  to quit and h for help.

If you are only interested in a subdirectory of this /project space and do not want to navigate the whole tree in the ncurse user interface, use

A complete manual page is available with the man duc command.

=== Graphical user interface ===

Note that when the login node is especially busy or if you have an especially large number of files in your /project space, the graphical interface mode can be slow and choppy. For a better experience, you can read the section below to run diskusage_explorer on your own machine.

Note that we recommend the use of the standard text-based ncurse mode on our cluster login nodes but diskusage_explorer does also include a nice graphical user interface (GUI).

First, make sure that you are connected to the cluster in such a way that SSH is capable of correctly displaying GUI applications. You can then use a graphical interface by means of the command,

You can navigate the folders with the mouse and still type c to toggle between the size of the files and the number of files.

=== Browse faster on your own machine ===

First, install the diskusage_explorer software on your local machine and then, still on your local machine, download the SQLite file from your cluster and run duc.

rsync -v --progress username@beluga.calculcanada.ca:/project/.duc_databases/def-professor.sqlite  .
duc gui -d ./def-professor.sqlite  /project/def-professor

This immediately leads to a smoother and more satisfying browsing experience.

== Space and file count usage per user on Cedar ==

On Cedar, it is possible for any member of a group to run diskusage_report with the following options --per_user and --all_users to have the breakdown per user. The first option displays only heavy users. In other terms, members of the group who have more files and/or occupy more space. When both options are used, the command gives the breakdown for all members of the group. This is a handy command that helps to identify the users within a group who have more files and/or a large amount of data and ask them to better manage their data by reducing their file count usage for example.

In the following example, user user01 runs the command and gets the following output:

[user01@cedar1 ~]$ diskusage_report --per_user --all_users
                             Description                Space           # of files
                     /home (user user01)             109k/50G              12/500k
                  /scratch (user user01)             4000/20T              1/1000k
                 /project (group user01)              0/2048k               0/1025
          /project (group def-professor)            9434G/10T            497k/500k

Breakdown for project def-professor (Last update: 2023-05-02 01:03:10)
           User      File count                 Size             Location
-------------------------------------------------------------------------
         user01           28313             4.00 GiB              On disk
         user02           11926             3.74 GiB              On disk
         user03           14507          6121.03 GiB              On disk
         user04            4010           377.86 GiB              On disk
         user05          125929           262.75 GiB              On disk
         user06          201099            60.51 GiB              On disk
         user07           84806          1721.33 GiB              On disk
         user08           26516           947.23 GiB              On disk
          Total          497106          9510.43 GiB              On disk

Breakdown for nearline def-professor (Last update: 2023-05-02 01:01:30)
           User      File count                 Size             Location
-------------------------------------------------------------------------
         user03               5          1197.90 GiB     On disk and tape
          Total               5          1197.90 GiB     On disk and tape

This group has 8 users and the above output shows clearly that at least 4 of them have a large number of files for a small amount of data:

           User      File count                 Size             Location
-------------------------------------------------------------------------
         user01           28313             4.00 GiB              On disk
         user02           11926             3.74 GiB              On disk
         user05          125929           262.75 GiB              On disk
         user06          201099            60.51 GiB              On disk