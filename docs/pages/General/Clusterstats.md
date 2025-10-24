---
title: "Clusterstats"
url: "https://docs.alliancecan.ca/wiki/Clusterstats"
category: "General"
last_modified: "2021-03-30T20:23:25Z"
page_id: 16419
display_title: "Clusterstats"
---

## Cluster Information {#cluster_information}

Clusterstats is a custom utility which displays information on partitions, nodes, jobs, your account, your group(s), and your priority.

To run clusterstats, just type the command on a cluster.

Clusterstats may take a few minutes to update and get fresh data, or it may use some recently-cached data.

    [✔] Loading node information (success, loaded cached version that is 2 min old)
    [✔] Loading job information (success, loaded cached version that is 2 min old)
    [✔] Loading share information (success, loaded cached version that is 1 min old)

Once the data is loaded, the main menu will appear asking if you would like information about your user, your group, or the state of the cluster. You can scroll up and down with the arrow keys, and make a selection with the \"Enter\" key. You can move back a level by selecting \"Back\" and quit the program by selecting \"Quit\".

    Information on? (Use arrow keys, press Enter to select)
    ‣ User
      Group
      Cluster
      Quit

### Information on the Cluster {#information_on_the_cluster}

You will be asked a number of questions about what part of the cluster you wish to see, and what type of information do you wish to display. Once you do, you will see a table listing the resources grouped by node type, and by the maximum run-time of jobs those nodes allow. Notice that the longer the run-time of a job is, the fewer resources are available to it.

    Information on? Cluster
    Please select on which part of the cluster would you like more information? CPU, (highmem or large) more than 12 GB of RAM per Core
    Information on ? Jobs/Partitions/Nodes for whole node jobs
    Please select the information you would like to display? Nodes

              This table shows all available resources in the partition. 
              A resource that is available to run 0-24 hour jobs 
              will show up in the (0-3),(3-12) and (12-24) columns.

    ┌───────────────────────┬─────────────┬────────┬─────────┬──────────┬─────────┬─────────┬──────────┐
    │ cpularge_bynode       │ interactive │ 0-3 hr │ 3-12 hr │ 12-24 hr │ 1-3 day │ 3-7 day │ 7-28 day │
    ├───────────────────────┼─────────────┼────────┼─────────┼──────────┼─────────┼─────────┼──────────┤
    │ Total (Nodes)         │ 2           │ 50     │ 50      │ 50       │ 35      │ 17      │ 7        │
    │   cpu=32, Mem=3095000 │ 0           │ 4      │ 4       │ 4        │ 4       │ 1       │ 1        │
    │   cpu=32, Mem=1547000 │ 0           │ 24     │ 24      │ 24       │ 16      │ 8       │ 3        │
    │   cpu=32, Mem=515000  │ 2           │ 22     │ 22      │ 22       │ 15      │ 8       │ 3        │
    │ Idle (Nodes)          │ 2           │ 0      │ 0       │ 0        │ 0       │ 0       │ 0        │
    │   cpu=32, Mem=515000  │ 2           │ 0      │ 0       │ 0        │ 0       │ 0       │ 0        │
    │ Running (Nodes)       │ 0           │ 46     │ 46      │ 46       │ 34      │ 17      │ 7        │
    │   cpu=32, Mem=3095000 │ 0           │ 3      │ 3       │ 3        │ 3       │ 1       │ 1        │
    │   cpu=32, Mem=1547000 │ 0           │ 21     │ 21      │ 21       │ 16      │ 8       │ 3        │
    │   cpu=32, Mem=515000  │ 0           │ 22     │ 22      │ 22       │ 15      │ 8       │ 3        │
    │ Down (Nodes)          │ 0           │ 4      │ 4       │ 4        │ 1       │ 0       │ 0        │
    │   cpu=32, Mem=3095000 │ 0           │ 1      │ 1       │ 1        │ 1       │ 0       │ 0        │
    │   cpu=32, Mem=1547000 │ 0           │ 3      │ 3       │ 3        │ 0       │ 0       │ 0        │
    └───────────────────────┴─────────────┴────────┴─────────┴──────────┴─────────┴─────────┴──────────┘

**cpu=32, Mem=515000** means that the nodes in that row have 32 cpu cores and 515,000MiB of system memory (RAM).

An example for GPU nodes:

    Please select on which part of the cluster would you like more information? GPU
    Information on ? Jobs/Partitions/Nodes for whole node jobs
    Please select the information you would like to display? Nodes

              This table shows all available resources in the partition. 
              A resource that is available to run 0-24 hour jobs 
              will show up in the (0-3),(3-12) and (12-24) columns.

    ┌───────────────────────────────┬─────────────┬────────┬─────────┬──────────┬─────────┬─────────┬──────────┐
    │ gpubase_bynode                │ interactive │ 0-3 hr │ 3-12 hr │ 12-24 hr │ 1-3 day │ 3-7 day │ 7-28 day │
    ├───────────────────────────────┼─────────────┼────────┼─────────┼──────────┼─────────┼─────────┼──────────┤
    │ Total (Nodes)                 │ 2           │ 336    │ 336     │ 270      │ 204     │ 120     │ 60       │
    │   p100:4 , cpu=24, Mem=128000 │ 2           │ 112    │ 112     │ 88       │ 64      │ 32      │ 16       │
    │   p100l:4, cpu=24, Mem=257000 │ 0           │ 32     │ 32      │ 28       │ 24      │ 12      │ 6        │
    │   v100l:4, cpu=32, Mem=192000 │ 0           │ 192    │ 192     │ 154      │ 116     │ 76      │ 38       │
    │ Idle (Nodes)                  │ 1           │ 2      │ 2       │ 1        │ 1       │ 1       │ 1        │
    │   p100:4 , cpu=24, Mem=128000 │ 1           │ 0      │ 0       │ 0        │ 0       │ 0       │ 0        │
    │   p100l:4, cpu=24, Mem=257000 │ 0           │ 2      │ 2       │ 1        │ 1       │ 1       │ 1        │
    │ Running (Nodes)               │ 0           │ 315    │ 315     │ 254      │ 194     │ 116     │ 57       │
    │   p100:4 , cpu=24, Mem=128000 │ 0           │ 104    │ 104     │ 83       │ 62      │ 31      │ 16       │
    │   p100l:4, cpu=24, Mem=257000 │ 0           │ 26     │ 26      │ 23       │ 20      │ 11      │ 5        │
    │   v100l:4, cpu=32, Mem=192000 │ 0           │ 185    │ 185     │ 148      │ 112     │ 74      │ 36       │
    │ Down (Nodes)                  │ 1           │ 19     │ 19      │ 15       │ 9       │ 3       │ 2        │
    │   p100:4 , cpu=24, Mem=128000 │ 1           │ 8      │ 8       │ 5        │ 2       │ 1       │ 0        │
    │   p100l:4, cpu=24, Mem=257000 │ 0           │ 4      │ 4       │ 4        │ 3       │ 0       │ 0        │
    │   v100l:4, cpu=32, Mem=192000 │ 0           │ 7      │ 7       │ 6        │ 4       │ 2       │ 2        │
    └───────────────────────────────┴─────────────┴────────┴─────────┴──────────┴─────────┴─────────┴──────────┘

**v100l:4, cpu=32, Mem=192000** means that the nodes in that row have 4 v100 GPUS with 32 GB of GPU RAM (v100l) as well as 32 CPU cores and 192,000 MiB of system memory (RAM).

### Information on your Group(s) {#information_on_your_groups}

When you choose \"Information on Group\", you will be prompted to select one of the accounting groups that you belong to. You will see a table of all the users who are members of that accounting group, the group\'s share of the cluster and how much the group has recently used it, each member\'s share of the group and how much that member has recently used it, and Fairshare values derived therefrom. The group\'s LevelFS is the group\'s share of the cluster divided by the group\'s use. Fairshare is the main component of the priority of any job.

    Information on? Group
    Information on Job ? def-kamil-ab_cpu
    ┌──────────────────┬──────────┬───────────┬───────────┬──────────┬─────────┬─────────┬───────────────┐
    │ Account          │ User     │ Group     │ Group     │ Group    │ Users's │ Users's │ Users's       │
    │                  │          │ Share     │ Used      │ LevelFS  │ Share   │ Used    │ Fairshare     │
    │                  │          │ % Cluster │ % Cluster │          │ % Group │ % Group │ Using Account │
    ├──────────────────┼──────────┼───────────┼───────────┼──────────┼─────────┼─────────┼───────────────┤
    │ def-kamil-ab_cpu │ kamil    │ SLEEPING  │ 0.0       │ SLEEPING │ 50.0    │ 100.0   │ SLEEPING      │
    │ def-kamil-ab_cpu │ tmcguire │ SLEEPING  │ 0.0       │ SLEEPING │ 50.0    │ 0.0     │ SLEEPING      │
    └──────────────────┴──────────┴───────────┴───────────┴──────────┴─────────┴─────────┴───────────────┘

In this example user, kamil has 50% of the group\'s share but used 100% of the resources used by the group. The default group def-kamil-ab_cpu has used almost zero resources and is currently inactive.

Shares of active default groups are set to the (unallocated resources/number of active default groups). Inactive default groups get no share and are labelled as SLEEPING, when a group member submits a job the group is soon classified as active)

### Information on the User {#information_on_the_user}

If you choose \"Information on User\" you will be asked to select \"Account\" or \"Jobs\".

#### User Account table {#user_account_table}

Select \"Account\" and you will get information in a table for all the groups you are a member of, as described in the Groups section above, but you will not see the other group members.

    Information on? User
    Information on ? Account
    ┌──────────────────┬───────────┬───────────┬──────────┬─────────┬─────────┬───────────────┐
    │ Account          │ Group     │ Group     │ Group    │ kamil's │ kamil's │ kamil's       │
    │                  │ Share     │ Used      │ LevelFS  │ Share   │ Used    │ Fairshare     │
    │                  │ % Cluster │ % Cluster │          │ % Group │ % Group │ Using Account │
    ├──────────────────┼───────────┼───────────┼──────────┼─────────┼─────────┼───────────────┤
    │ def-kamil-ab_cpu │ SLEEPING  │ 0.0       │ SLEEPING │ 50.0    │ 100.0   │ SLEEPING      │
    │ def-kamil-ab_gpu │ SLEEPING  │ 0.0       │ SLEEPING │ 50.0    │ 100.0   │ SLEEPING      │
    │ rrg-kamil_gpu    │ 0.0495    │ 0.0815    │ 0.606848 │ 6.25    │ 75.8338 │ 0.325922      │
    └──────────────────┴───────────┴───────────┴──────────┴─────────┴─────────┴───────────────┘

#### User Jobs options {#user_jobs_options}

If you select \"Jobs\", and you have any jobs running or pending, you will be prompted to select a particular job you wish more information on. For a given job, you can select \"Basic\", \"Report\", or \"Output of the scontrol command\". Selecting \"Basic\" will print a one-line summary giving the job\'s state (pending, running, or recently completed), its partition, its priority, and if pending, its rank.

    Information on ? Basic
    Job:46460857 state: pending partition: cpubase_bycore_b4 priority: 1618298
        This job is ranked 1517 of 7825 in terms of priority

The rank has the following meaning: The nodes that can potentially run the job can also run 7825 other jobs. When all these jobs are ranked by priority from highest to lowest, this job is 1517th.

Selecting \"Report\" will show the nodes that your job can run on and the state of those nodes.

    Information on ? Report
    Job 65066247:
        This pending job belongs to user kamil, accounting group rrg-kamil_gpu in partition gpubase_bynode_b5
        Nodes that can possibly run the job:
          Total: 120 Busy: 116 Down: 3 Idle: 1
            Node Type (p100:4, cpu=24, Mem=128000):  Total 32 Down 1 Idle 0
            Node Type (p100l:4, cpu=24, Mem=257000):  Total 12 Down 0 Idle 1
            Node Type (v100l:4, cpu=32, Mem=192000):  Total 76 Down 2 Idle 0
          This job is ranked 46 of 3737 in terms of priority on these nodes

Selecting \"Output of the scontrol command\" will display more detailed diagnostics obtained from `scontrol show job`.
