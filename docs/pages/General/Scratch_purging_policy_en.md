---
title: "Scratch purging policy/en"
url: "https://docs.alliancecan.ca/wiki/Scratch_purging_policy/en"
category: "General"
last_modified: "2025-07-25T17:33:02Z"
page_id: 6041
display_title: "Scratch purging policy"
---

`<languages />`{=html}

*Parent page: [Storage and file management](https://docs.alliancecan.ca/Storage_and_file_management "Storage and file management"){.wikilink}*

The scratch filesystem on our clusters is intended as temporary, fast storage for data being used during job execution. Data needed for long-term storage and reference should be kept in either [`/project`](https://docs.alliancecan.ca/Project_layout "/project"){.wikilink} or other archival storage areas. In order to ensure adequate space on scratch, files older than 60 days are periodically deleted according to the policy outlined in this page. Note that the purging of a file is based on its age, not its location within scratch; simply moving a file from one directory in scratch to another directory in scratch will not in general prevent it from being purged.

**Please note: on Graham and Nibi, the scratch filesystem does not explicitly expire, but uses quota to enforce transient use.** These clusters share a single storage system, and on their /scratch, it works like this:

- If your usage is under 1T, you never have to do anything (no expiration, etc).
- when you exceed this \"soft limit\", you trigger an over-quota condition, which includes a 60-day grace period.
- when the grace period expires, over-quota enforcement starts, disallowing any new allocation. (This implies that your files at that point are still accessible, but adding files or increasing file sizes will fail with an error.)
- to resolve this, you must bring your usage under the soft limit (which \"resets\" the clock).
- there is also a hard quota of 20T.

This quota-based mechanism has the advantage of being fully user-controlled, and avoids the case where specific files in a collection disappear due to slightly different age.

## Expiration procedure {#expiration_procedure}

The scratch filesystem is checked at the end of the month for files which will be candidates for expiry on the 15th of the following month. On the first day of the month, a login message is posted and a notification e-mail is sent to all users who have at least one file which is a candidate for purging and containing the location of a file which lists all the candidates for purging. You will thus have two weeks to make arrangements to copy data to your project space or some other location if you wish to save the data in question.

On the 12th of the month, a final notification e-mail will be sent with an updated assessment of candidate files for expiration on the 15th, giving you 72 hours to make arrangements for moving these files. At the end of the day on the 15th, any remaining files on the scratch filesystem for which both the `ctime` and the `atime` are older than 60 days will be deleted. Please remember that the e-mail reminders and login notice are a courtesy offered to our users, whose ultimate responsibility it is to keep files older than 60 days from being located in the scratch space.

Note that simply copying or using the `rsync` command to displace your files will update the `atime` for the original data on scratch, making them ineligible for deletion. Once you have put the data in another location please delete the original files and directories in scratch instead of depending on the automatic purging.

## How/where to check which files are slated for purging {#howwhere_to_check_which_files_are_slated_for_purging}

- On Cedar, Beluga, and Narval clusters go to the **/scratch/to_delete/** path and look for a file with your name.
- On Niagara go to **/scratch/t/to_delete/** (symlink to /scratch/t/todelete/current)

</pre>

The file will contain a list of filenames with full path, possibly other information about atime, ctime, size, etc. It will be updated only on the 1st and the 12th of each month. If a file with your name is there, it means you have candidates slated for purging, otherwise there is nothing to worry about that month.

If you access/read/move/delete some of the candidates between the 1st and the 11th, there won\'t be any changes in the assessment until the 12th.

If there was an assessment file up until the 11th, but no longer on the 12th, it\'s because you don\'t have anything to be purged anymore.

If you access/read/move/delete some or the candidates after the 12th, then you have to check yourself to confirm your files won\'t be purged on the 15th (see below)

## How do I check the age of a file? {#how_do_i_check_the_age_of_a_file}

We define a file\'s age as the most recent of:

- the access time (`atime`) and
- the change time (`ctime`).

You can find the `ctime` of a file using

while the `atime` can be obtained with the command

We do not use the modify time (`mtime`) of the file because it can be modified by the user or by other programs to display incorrect information.

Ordinarily, simple use of the `atime` property would be sufficient, as it is updated by the system in sync with the `ctime`. However, userspace programs are able to alter `atime`, potentially to times in the past, which could result in early expiration of a file. The use of `ctime` as a fallback guards against this undesirable behaviour.

## Abuse

This method of tracking file age does allow for potential abuse by periodically running a recursive `touch` command on your files to prevent them from being flagged for expiration. Our staff have methods for detecting this and similar tactics to circumvent the purging policy. Users who employ such techniques will be contacted and asked to modify their behaviour, in particular to move the \"retouched\" data from scratch to a more appropriate location.

## How to safely copy a directory with symlinks {#how_to_safely_copy_a_directory_with_symlinks}

In most cases, `cp` or `rsync` will be sufficient to copy data from scratch to project. But if you have symbolic links in scratch, copying them will cause problems since they will still point to scratch. To avoid this, you can use `tar` to make an archive of your files on scratch, and extract this archive in your project. You can do this in one go:

``` bash
cd /scratch/.../your_data
mkdir project/.../your_data
tar cf - ./* | (cd /project/.../your_data && tar xf -)
```
