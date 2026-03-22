---
title: "Scratch purging policy/en"
url: "https://docs.alliancecan.ca/wiki/Scratch_purging_policy/en"
category: "General"
last_modified: "2026-03-19T18:59:54Z"
page_id: 6041
display_title: "Scratch purging policy"
---

Parent page: Storage and file management

On our clusters, the /scratch filesystem is intended for temporary, fast storage for data being used during job execution.  Data needed for long-term storage and reference should be kept in either /project or other archival storage areas; interested readers can obtain details on the nature of /scratch and other filesystems. In order to ensure adequate space on /scratch, files older than 60 days are periodically deleted according to the policy outlined in this page.  Note that the purging of a file is based on its age, not its location within /scratch; simply moving a file from one directory in /scratch to another directory in /scratch will not in general prevent it from being purged.

Please note: The method for Nibi is different from that of the other clusters (see below).

The quota-based mechanism has the advantage of being fully user-controlled, and avoids the case where specific files in a collection disappear due to slightly different age.

==Expiration procedure==

===Fir, Narval, Rorqual, Trillium===
The /scratch filesystem is checked at the end of the month for files which will be candidates for expiry on the 15th of the following month. On the first day of the month, a login message is posted and a notification e-mail is sent to all users who have at least one file which is a candidate for purging; the notification also contains  the location of a file where all the candidates for purging are listed. You will thus have two weeks to make arrangements to copy data to your /project space or some other location if you wish to save it.

On the 12th of the month, a final notification e-mail is sent with an updated assessment of candidate files for expiration on the 15th, giving you 72 hours to make arrangements to move them. At the end of the day on the 15th, any remaining files on the /scratch filesystem for which both the ctime and the atime timestamps  are older than 60 days are deleted. Please remember that the reminders and notifications are a courtesy offered to our users, whose ultimate responsibility it is to keep files older than 60 days out of the /scratch space.
Once you have put the data in another location please delete the original files and directories in  /scratch instead of depending on automatic purging.

===Nibi===
* If your usage remains under a soft limit (currently set at 1TB) you never have to do anything. There is no time limit, and nothing is automatically deleted.
* Exceeding this soft limit triggers an over-quota condition, which includes a 60-day grace period.
* When the grace period expires, over-quota enforcement starts.  Your files are still accessible, but adding files or increasing file sizes will fail with an error.
* In order to add files or increase their size you must first bring your usage under the soft limit, which resets the 60-day counter.
* There is also a hard limit of 20TB, which means that even within the 60-day grace period, you cannot store more than 20TB of data in /scratch.

==How to check which files are scheduled for purging==

* On Fir and Narval clusters go to the /scratch/to_delete/ path  and look for a file with your name.
* On Trillium go to /scratch/t/to_delete/ (symlink to /scratch/t/todelete/current).

The file will contain a list of filenames with their full path, and possibly other information about atime, ctime, size, etc. It will be updated only on the 1st and the 12th of each month. If a file with your name is there, it means you have candidates for purging, otherwise there is nothing to purge that month.

If you access/read/move/delete some of the candidates between the 1st and the 11th, there won't be any changes in the assessment until the 12th.

If there was an assessment file up until the 11th, but no longer on the 12th, it's because you don't have anything to be purged anymore.

If you access/read/move/delete some or the candidates after the 12th, then you have to check for yourself to confirm your files won't be purged on the 15th (see below).

==How to check the age of a file?==
A file's age is determined by the younger of either
* the access time (atime) or
* the change time (ctime).

Both these timestamps are printed by
 [name@server ~]$ stat

The latest modification time (mtime) is not used because it can be backdated to a distant time in the past by you or by programs that you use (such as version-control or build-system software like Git or Make).

==Abuse==
This method of tracking file age does allow for potential abuse by periodically running a recursive touch command on your files to prevent them from being flagged for expiration. Our staff have methods for detecting this and similar tactics aimed to circumvent the purging policy. Users who employ such techniques will be asked to move the offending data from /scratch to another location.

==How to safely copy a directory with symlinks==

In most cases, cp or rsync will be sufficient to copy data from /scratch to /project. But if you have symbolic links in /scratch, copying them will cause problems since they will still point to /scratch. To avoid this, you can use tar to make an archive of your files on /scratch, and extract this archive in /project as such

cd /scratch/.../your_data
mkdir project/.../your_data
tar -cf - ./* | tar -C /project/.../your_data -xf -