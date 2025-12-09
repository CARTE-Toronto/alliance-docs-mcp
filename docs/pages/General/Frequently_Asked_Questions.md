---
title: "Frequently Asked Questions/en"
url: "https://docs.alliancecan.ca/wiki/Frequently_Asked_Questions/en"
category: "General"
last_modified: "2024-07-15T19:47:35Z"
page_id: 5091
display_title: "Frequently Asked Questions"
---

__TOC__

== Forgot my password ==
To reset your password for any Alliance national cluster, visit https://ccdb.computecanada.ca/security/forgot. Please note that you will not be able to reset your password until your first role gets approved by our staff.

== Copy and paste ==
In the Linux terminal, you can't use [Ctrl]+C to copy text,
because [Ctrl]+C means "Cancel" or "Interrupt" and stops the running program.

Instead you can use [Ctrl]+[Insert] to copy and [Shift]+[Insert] to paste
in most cases under Windows and Linux, depending on your terminal program.
Users of macOS can continue to use [Cmd]+C and [Cmd]+V to copy and paste.

Depending on which terminal software you are using, you simply need to select
the text to copy it into the clipboard, and you can paste from the clipboard
by using either the right-click or middle-click (the default setting can vary).

==Text file line endings ==
For historical reasons, Windows and most other operating systems, including Linux and OS X, disagree on the convention that is used to denote the end of a line in a plain text ASCII file. Text files prepared in a Windows environment will therefore have an additional invisible "carriage return" character at the end of each line and this can cause certain problems when reading this file in a Linux environment. For this reason you should either consider creating and editing your text files on the cluster itself using standard Linux text editors like emacs, vim and nano or, if you prefer Windows, to then use the command dos2unix  on the cluster login node to convert the line endings of your file to the appropriate convention.

== Saving files is slow in my editor ==

=== Emacs ===

Emacs uses the fsync system call when saving files to reduce the risk of losing data in the case of a system crash. This extra reliability comes at a cost: sometimes it can take several seconds to save even a small file when writing to a shared filesystem (e.g., home, scratch, project) on one of the clusters. If you find that your work is impacted by slow file saves, you can add the following line to your ~/.emacs file to increase performance:

(setq write-region-inhibit-fsync t)

More about this setting here: Customize save in Emacs

== Moving files across the project, scratch and home filesystems ==
On our general purpose clusters, the scratch and home filesystems have quotas that are per-user, while the project filesystem has quotas that are per-project. Because the underlying implementation of quotas on the Lustre filesystem
is currently based on group ownership of files, it is important to ensure that the files have the right group. On the scratch and home filesystems, the correct group is typically the group with the same name as your username. On the project filesystem, group name should follow the pattern prefix-piusername where prefix is typically one of def, rrg, rpp.

=== Moving files between scratch and home filesystems ===
Since the quotas of these two filesystems are based on your personal group, you should be able to move files across the two using

=== Moving files from scratch or home filesystems to project ===
If you want to move files from your scratch or home space into a project space, you should not use the mv command. Instead, we recommend using the regular cp, or the rsync command.

It is very important to run cp and rsync correctly to ensure that the files copied over to the project space have the correct group ownership. With cp, do not use the archive -a option. And when using rsync, make sure you specify the --no-g --no-p options, like so:

Once the files are copied, you can then delete them from your scratch space.

=== Moving files from project to scratch or home filesystems ===
If you want to move files from your project into your scratch or home space, you should not use the mv command. Instead, we recommend using the regular cp, or the rsync command.

It is very important to run cp and rsync correctly to ensure that the files copied over to the project space have the correct group ownership. With cp, do not use the archive -a option. And when using rsync, make sure you specify the --no-g --no-p options, like so:

=== Moving files between two project spaces ===
If you want to move files between two project spaces, you should not use the mv command. Instead, we recommend using the regular cp, or the rsync command.

It is very important to run cp or rsync correctly to ensure that the files copied over have the correct group ownership. With cp, do not use the archive -a option. And when using rsync, make sure you specify the --no-g --no-p options, like so:

Once you have copied your data over, please delete the old data.

== Disk quota exceeded error on /project filesystems ==
:Also see: Project layout
Some users have seen this message or some similar quota error on their /project folders. Other users have reported obscure failures while transferring files into their /project folder from another cluster. Many of the problems reported are due to bad file ownership.

Use diskusage_report to see if you are at or over your quota:

[ymartin@cedar5 ~]$ diskusage_report
                             Description                Space           # of files
                     Home (user ymartin)             345M/50G            9518/500k
                  Scratch (user ymartin)              93M/20T           6532/1000k
                 Project (group ymartin)          5472k/2048k            158/5000k
            Project (group def-zrichard)            20k/1000G              4/5000k

The example above illustrates a frequent problem: /project for user ymartin contains too much data in files belonging to group ymartin. The data should instead be in files belonging to def-zrichard. To see the project groups you may use, run the following command:
 stat -c %G $HOME/projects/*/

Note the two lines labelled Project.
*Project (group ymartin) describes files belonging to group ymartin, which has the same name as the user. This user is the only member of this group, which has a very small quota (2048k).
*Project (group def-zrichard) describes files belonging to a project group. Your account may be associated with one or more project groups, and they will typically have names like def-zrichard, rrg-someprof-ab, or rpp-someprof.

In this example, files have somehow been created belonging to group ymartin instead of group def-zrichard. This is neither the desired nor the expected behaviour.

By design, new files and directories in /project will normally be created belonging to a project group. The main reasons why files may be associated with the wrong group are
* files were moved from /home to /project with the mvcommand; to avoid this, see  advice above;
* files were transferred from another cluster using rsync or scp with an option to preserve the original group ownership. If you have a recurring problem with ownership, check the options you are using with your file transfer program;
* you have no setgid bit set on your Project folders.

=== How to fix the problem ===
If you already have data in your /project directory with the wrong group ownership, you can use the find to display those files:
 lfs find ~/projects/*/ -group $USER

Next, change ownership from $USER to the project group, for example:
 chown -h -R $USER:def-professor -- ~/projects/def-professor/$USER/

Then, set the setgid bit on all directories (for more information, see Group ID) to ensure that newly created files will inherit the directory's group membership, for example:
 lfs find ~/projects/def-professor/$USER -type d -print0 | xargs -0 chmod g+s

Finally, verify that project space directories have correct permissions set
 chmod 2770 ~/projects/def-professor/
 chmod 2700 ~/projects/def-professor/$USER

=== Another explanation ===
Each file in Linux belongs to a person and a group at the same time. By default, a file you create belongs to you, user username, and your group, named the same username. That is it is owned by username:username. Your group is created at the same time your account was created and you are the only user in that group.

This file ownership is good for your home directory and the scratch space, as shown hereː

                              Description                Space           # of files
                      Home (user username)              15G/53G             74k/500k
                   Scratch (user username)           1522G/100T            65k/1000k
                  Project (group username)            34G/2048k             330/2048
             Project (group def-professor)            28k/1000G               9/500k

The quota is set for these for a user username.

The other two lines are set for groups username and def-professor in Project space. It is not important what users own the files in that space, but the group the files belong to determines the quota limit.

You see that files that are owned by username group (your default group) have very small limit in the project space, only 2MB, and you already have 34 GB of data that is owned by your group (your files). This is why you cannot write more data there. Because you are trying to place data there owned by a group that has very little allocation there.

On the other hand, the allocation for the group def-professor, your professor's group, does not use almost any space and has 1 TB limit. The files that can be put there should have username:def-professor ownership.

Now, depending on how you copy your files, what software you use, that software either will respect the ownership of the directory and apply the correct group, or it may insist on retaining the ownership of the source data. In the latter case, you will have a problem like you have now.

Most probably your original data belongs to username:username, properly, upon moving it, it should belong to username:def-professor, but your software probably insists on keeping the original ownership and this causes the problem.

== sbatch: error: Batch job submission failed: Socket timed out on send/recv operation ==

You may see this message when the load on the Slurm manager or scheduler process is too high. We are working both to improve Slurm's tolerance of that and to identify and eliminate the sources of load spikes, but that is a long-term project. The best advice we have currently is to wait a minute or so. Then run squeue -u $USER and see if the job you were trying to submit appears: in some cases the error message is delivered even though the job was accepted by Slurm. If it doesn't appear, simply submit it again.

== Why are my jobs taking so long to start? ==
You can see why your jobs are in the PD (pending) state by running the squeue -u  command on the cluster.
The (REASON) column typically has the values Resources or Priority.
* Resourcesː The cluster is simply very busy and you will have to be patient or perhaps consider if you can submit a job that asks for fewer resources (e.g. CPUs/nodes, GPUs, memory, time).
*  Priorityː Your job is waiting to start due to its lower priority. This is because you and other members of your research group have been over-consuming your fair share of the cluster resources in the recent past, something you can track using the command sshare as explained in Job scheduling policies. The LevelFS column gives you information about your over- or under-consumption of cluster resources: when LevelFS is greater than one, you are consuming fewer resources than your fair share, while if it is less than one you are consuming more. The more you overconsume resources, the closer the value gets to zero and the more your pending jobs decrease in priority. There is a memory effect to this calculation so the scheduler gradually "forgets" about any potential over- or under-consumption of resources from months past. Finally, note that the value of LevelFS is unique to the specific cluster.

== Why do my jobs show "Nodes required for job are DOWN, DRAINED or RESERVED for jobs in higher priority partitions" or "ReqNodeNotAvailable"? ==

One of these strings may appear in the "Reason" field of squeue output for a waiting job, and is new to Slurm 19.05.
They mean one or more of the nodes Slurm considered for the job are down, or deliberately taken offline,
or are being reserved for other jobs.  On a large busy cluster, there will almost always be such nodes.  The messages mean
effectively the same thing as the reason "Resources" that appeared in Slurm version 17.11.  However, these are not
error messages; jobs submitted are actually in the queue and will eventually be processed.

== How accurate is START_TIME in squeue output? ==
We don't show the start time by default with squeue, but it can be printed with an option.  The start times Slurm forecasts depend on rapidly changing conditions, and are therefore not very useful.

Slurm computes START_TIME for high-priority pending jobs. These expected start times are computed from currently available information:
* What resources will be freed by running jobs that complete; and
* what resources will be needed by other, higher-priority jobs waiting to run.

Slurm invalidates these future plans:
* if jobs end early, changing which resources become available; and
* if prioritization changes, due to submission of higher-priority jobs or cancellation of queued jobs for example.

On our general purpose clusters, new jobs are submitted about every five seconds, and 30-50% of jobs end early,
so Slurm often discards and recomputes its future plans.

Most waiting jobs have a START_TIME of "N/A", which stands for "not available", meaning Slurm is not attempting to project a start time for them.

For jobs which are already running, the start time reported by squeue is perfectly accurate.

==What are the .core files that I find in my directory?==

In some instances a program which crashes or otherwise exits abnormally will leave behind a binary file, called a core file, containing a snapshot of the program's state at the moment that it crashed, typically with the extension ".core". While such files can be useful for programmers who are debugging the software in question, they are normally of no interest for regular users beyond the indication that something went wrong with the execution of the software, something already indicated by the job's output normally. You can therefore delete these files if you wish and add the line ulimit -c 0 to the end of your $HOME/.bashrc file to ensure that they are no longer created.

==How to fix library not found error==
When installing precompiled binary packages in your $HOME, they may fail with an error such as /lib64/libc.so.6: version 'GLIBC_2.18' not found at runtime. See Installing binary packages for how to fix this kind of issue.

== How do you handle sensitive research data? ==

The Alliance does not operate any cluster specifically designated for handling personal data, private data, or sensitive data, such as (for example) human clinical research data.

Our resources are all administered following best practices for shared research systems, and we devote considerable effort to ensuring data integrity, confidentiality, and availability.  However, none of the resources are certified as meeting specific security assurance levels which may be required for certain datasets.  Responsibility for data protection and data privacy rests ultimately with the researcher.  Please see Privacy and Data Protection Policy section 5.2, and Terms of Use paragraph 3.12, at your Agreements page.

See Data protection, privacy, and confidentiality for more on this topic.