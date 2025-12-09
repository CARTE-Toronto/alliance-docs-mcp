---
title: "Dar/en"
url: "https://docs.alliancecan.ca/wiki/Dar/en"
category: "General"
last_modified: "2024-07-15T20:45:30Z"
page_id: 11266
display_title: "Dar"
---

Parent page: Storage and file management

The dar (stands for Disk ARchive) utility was written from the ground up as a modern
replacement to the classical Unix tar tool. First released in 2002, dar is open
source, is actively maintained, and can be compiled on any Unix-like system.

Similar to tar,
dar supports full / differential / incremental backups. Unlike tar, each
dar archive includes a file index for fast file access and restore -- this is especially useful for large
archives! dar has built-in compression on a file-by-file basis, making it more resilient
against data corruption, and you can optionally tell it not to compress already highly compressed files
such as mp4 and gz. dar supports strong encryption,
can split archives at 1-byte resolution, supports extended file attributes, sparse files, hard and
symbolic (soft) links, can detect data corruption in both headers and saved data and recover with minimal
data loss, and has many other desirable features. On the dar page you can find a detailed feature-by-feature tar-to-dar comparison.

== Where to find dar ==

On our clusters, dar is available on /cvmfs.
With StdEnv/2020:

[user_name@localhost]$ which dar
/cvmfs/soft.computecanada.ca/gentoo/2020/usr/bin/dar
[user_name@localhost]$ dar --version
dar version 2.5.11, Copyright (C) 2002-2052 Denis Corbin
...

== Using dar manually ==

=== Basic archiving and extracting ===

Let's say, in the current directory you have a subdirectory test. To pack it into an archive,
you can type in the current directory:

[user_name@localhost]$ dar -w -c all -g test

This will create an archive file all.1.dar, where all is the base name and
1 is the slice number. You can break a single archive into multiple slices (below). You can
include multiple directories and files into an archive, e.g.

[user_name@localhost]$ dar -w -c all -g testDir1 -g testDir2 -g file1 -g file2

Please note that all paths should be relative to the current directory.

To list the archive's contents, use only the base name:

[user_name@localhost]$ dar -l all

To extract a single file into a subdirectory restore, use the base name and the file path:

[user_name@localhost]$ dar -R restore/ -O -w -x all -v -g test/filename

The flag -O will tell dar to ignore file ownership. Wrong ownership would be a
problem if you are restoring someone else's files and you are not root. However, even if you are
restoring your own files, dar will throw a message that you are doing this as non-root and
will ask you to confirm. To disable this warning, use -O. The flag -w will
disable a warning if restore/test already exists.

To extract an entire directory, type:

[user_name@localhost]$ dar -R restore/ -O -w -x all -v -g test

Similar to creating an archive, you can pass multiple directories and files by using multiple
-g flags. Note that dar does not accept Unix wild masks after -g.

==== A note about the Lustre filesystem ====

If the archived files are coming from a Lustre filesystem
(typically in /home, /project or /scratch
on our general-purpose compute clusters),
some extended attributes are saved automatically.
To see which extended attributes are assigned to each archived file, use the -alist-ea flag:

We can see strings like: Extended Attribute: [lustre.lov].
With this attribute, any file extraction to a location formatted in Lustre will still work as usual.
But if one tries to extract files to the node local storage
(also known as $SLURM_TMPDIR),
the extraction will show error messages like:
Error while adding EA lustre.lov : Operation not supported.

To avoid these error messages, the -u flag can be used to exclude a specific type of attribute,
while the "affected" files are still extracted. For example:

Another solution is to get rid of the lustre.lov attribute
while creating the archive with the same -u flag:

In conclusion, this is necessary only if you intend to extract files to a location not formatted in Lustre.

=== Incremental backups ===

You can create differential and incremental backups with dar, by passing the base name of
the reference archive with -A. For example, let's say on Monday you create a full backup
named monday:

[user_name@localhost]$ dar -w -c monday -g test

On Tuesday you modify some of the files and then include only these files into a new, incremental backup
named tuesday, using the monday archive as a reference:

[user_name@localhost]$ dar -w -A monday -c tuesday -g test

On Wednesday you modify more files, and at the end of the day you create a new backup named
wednesday, now using the tuesday archive as a reference:

[user_name@localhost]$ dar -w -A tuesday -c wednesday -g test

Now you have three files:

[user_name@localhost]$ ls *.dar
monday.1.dar     tuesday.1.dar    wednesday.1.dar

The file wednesday.1.dar contains only the files that you modified on Wednesday, but not the
files from Monday or Tuesday. Therefore, the command

[user_name@localhost]$ dar -R restore -O -x wednesday

will only restore files that were modified on Wednesday. To restore everything, you have to go through
all backups in the chronological order:

[user_name@localhost]$ dar -R restore -O -w -x monday      # restore the full backup
[user_name@localhost]$ dar -R restore -O -w -x tuesday     # restore the first incremental backup
[user_name@localhost]$ dar -R restore -O -w -x wednesday   # restore the second incremental backup

=== Limiting the size of each slice ===

To limit the maximum size of each slice in bytes, use the flag -s followed by a number and one of k/M/G/T. For example, for a 1340 MB archive, the command

[user_name@localhost]$ dar -s 100M -w -c monday -g test

will create 14 slices named monday.{1..14}.dar. To extract from all of these, use their base name:

[user_name@localhost]$ dar -O -x monday

== External scripts ==

One of our team members has written bash functions that can facilitate the use of dar. You can use these functions as inspiration to write your own scripts. See here for details.