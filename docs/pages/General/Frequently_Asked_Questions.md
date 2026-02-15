---
title: "Frequently Asked Questions/en"
url: "https://docs.alliancecan.ca/wiki/Frequently_Asked_Questions/en"
category: "General"
last_modified: "2026-02-06T20:01:02Z"
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