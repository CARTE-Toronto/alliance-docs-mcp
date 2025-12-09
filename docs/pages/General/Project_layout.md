---
title: "Project layout/en"
url: "https://docs.alliancecan.ca/wiki/Project_layout/en"
category: "General"
last_modified: "2025-10-17T15:14:01Z"
page_id: 4342
display_title: "Project layout"
---

`<languages />`{=html}

:   `<i>`{=html}Parent page: [Storage and file management](https://docs.alliancecan.ca/Storage_and_file_management "Storage and file management"){.wikilink}`</i>`{=html}
:   `<i>`{=html}See also: [ Disk quota exceeded error on /project filesystems](https://docs.alliancecan.ca/Frequently_Asked_Questions#Disk_quota_exceeded_error_on_.2Fproject_filesystems " Disk quota exceeded error on /project filesystems"){.wikilink}`</i>`{=html}

The project filesystem on our compute clusters is organized on the basis of `<i>`{=html}groups`</i>`{=html}. The normal method to access the project space is by means of symbolic links which exist in your home directory. These will have the form `$HOME/projects/group_name` apart from the clusters [Rorqual](https://docs.alliancecan.ca/Rorqual/en "Rorqual"){.wikilink} and [Trillium](https://docs.alliancecan.ca/Trillium "Trillium"){.wikilink} where the path will take the form `$HOME/links/projects/group_name`.

The permissions on the group space are such that it is owned by the principal investigator (PI) for this group and members have read and write permission on this directory. However by default a newly created file will only be readable by group members. If the group wishes to have writeable files, the best approach is to create a special directory for that, for example

followed by

For more on sharing data, file ownership, and access control lists (ACLs), see [Sharing data](https://docs.alliancecan.ca/Sharing_data "Sharing data"){.wikilink}.

The project space is subject to a default quota of 1 TB and 500,000 files per group and which can be increased up to 40 TB of space upon request to [Technical support](https://docs.alliancecan.ca/Technical_support "Technical support"){.wikilink}. Certain groups may have been awarded significantly higher quotas through the annual [Resource Allocation Competition](https://alliancecan.ca/en/services/advanced-research-computing/accessing-resources/resource-allocation-competition). In this case, you will already have been notified of your group\'s quota for the coming year. Note that this storage allocation is specific to a particular cluster and cannot normally be transferred to another cluster.

To check current usage and available disk space, use

In order to ensure that files which are copied or moved to a given project space acquire the appropriate group membership - and thus are counted against the expected quota - it can be useful to set the `setgid` bit on the directory in question. This will have the effect of ensuring that every new file and subdirectory created below the directory will inherit the same group as the ambient directory; equally so, new subdirectories will also possess this same `setgid` bit. However, existing files and subdirectories will not have their group membership changed - this should be done with the `chgrp` command - and any files moved to the directory will also continue to retain their existing group membership. You can set the `setgid` bit on a directory with the command

If you want to apply this command to the existing subdirectories of a directory, you can use the command

`xargs -0 chmod g+s}}`

More information on the `setgid` is available from this [page](https://en.wikipedia.org/wiki/Setuid#setuid_and_setgid_on_directories).

You can also use the command `newgrp` to modify your default group during an interactive session, for example

and then to copy any data to the appropriate project directory. This will only change your default group for this particular session however - at your next login you will need to reuse the `newgrp` command if you wish to change the default group again.

Note that if you are getting `<i>`{=html}disk quota exceeded`</i>`{=html} error messages (see [ Disk quota exceeded error on /project filesystems](https://docs.alliancecan.ca/Frequently_Asked_Questions#Disk_quota_exceeded_error_on_.2Fproject_filesystems " Disk quota exceeded error on /project filesystems"){.wikilink}), this may well be due to files being associated with the wrong group, notably your personal group, i.e. the one with the same name as your username and which has a quota of only 2 MB. To find and fix the group membership of such files you can use the command

`find ``<directory name>`{=html}` -group $USER -print0 | xargs -0 chgrp -h ``<group>`{=html}

where `<group>`{=html} is something like `def-profname`, thus a group with a reasonable quota of a terabyte or more.

### An explanatory example {#an_explanatory_example}

Imagine that we have a PI ("Sue") who has a sponsored user under her ("Bob"). Both Sue and Bob start with a directory structure that on the surface looks similar:

<div style="column-count:2;-moz-column-count:2;-webkit-column-count:2">

- `/home/sue/scratch` (symbolic link)
- `/home/sue/projects` (directory)
- `/home/bob/scratch` (symbolic link)
- `/home/bob/projects` (directory)

</div>

The scratch link points to a different location for Sue (`/scratch/sue`) and Bob (`/scratch/bob`).

If Bob\'s only role was the one sponsored by Sue, then Bob\'s `projects` directory would have the same contents as Sue\'s `projects` directory. Further, if neither Sue nor Bob have any other roles or projects with Alliance, then each one\'s `projects` directory would just contain one subdirectory, `def-sue`.

Each of `/home/sue/projects/def-sue` and `/home/bob/projects/def-sue` would point to the same location, `/project/``<some random number>`{=html}. This project directory is the best place for Sue and Bob to share data. They can both create directories in it, read it, and write to it. Sue for instance could do

`$ cd ~/projects/def-sue`\
`$ mkdir foo`

and Bob could then copy a file into the directory `~/projects/def-sue/foo`, where it will be visible to both of them.

If Sue were to get a RAC award with storage (as is often the case these days), both she and Bob would find that there is a new entry in their respective `projects` directory, something like

`~/projects/rrg-sue-ab`

They should use this directory to store and share data related to the research carried out under the RAC award.

For sharing data with someone who doesn\'t have a role sponsored by Sue\-\-- let\'s say Heather\-\-- the simplest thing to do is to change the file permissions so that Heather can read a particular directory or file. See [Sharing data](https://docs.alliancecan.ca/Sharing_data "Sharing data"){.wikilink} for more details. The best idea is usually to use ACLs to let Heather read a directory. Note that these filesystem permissions can be changed for almost any directory or file, not just those in your `project` space \-\-- you could share a directory in your `scratch` too, or just a particular subdirectory of `projects`, if you have several (a default one, one for a RAC, `<i>`{=html}etc.`</i>`{=html}). Best practice is to restrict file sharing to `/project` and `/scratch`.)

One thing to keep in mind when sharing a directory is that Heather will need to be able to descend the entire filesystem structure down to this directory and so she will need to have read and execute permission on each of the directories between `~/projects/def-sue` and the directory containing the file(s) to be shared. We have implicitly assumed here that Heather has an account on the cluster but you can even share with researchers who don\'t have a Alliance account using a [ Globus shared endpoint](https://docs.alliancecan.ca/Globus#Globus_Sharing " Globus shared endpoint"){.wikilink}.

If Heather is pursuing a serious and ongoing collaboration with Sue then it may naturally make sense for Sue to sponsor a role for Heather, thereby giving Heather access similar to Bob\'s, described earlier.

To summarize:

- `scratch` space is for (private) temporary files
- `home` space is normally for small amounts of relatively private data (e.g. a job script),
- Shared data for a research group should normally go in that group\'s `project` space, as it is persistent, backed up, and fairly large (up to 40 TB, or more with a RAC).
