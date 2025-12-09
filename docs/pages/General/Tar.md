---
title: "Tar/en"
url: "https://docs.alliancecan.ca/wiki/Tar/en"
category: "General"
last_modified: "2019-07-03T12:59:08Z"
page_id: 1577
display_title: "Tar/en"
---

Archiving means creating one file that contains a number of smaller files within it. Archiving data can improve the efficiency of file storage, and of file transfers. It is faster for the secure copy protocol (scp), for example, to transfer one archive file of a reasonable size than thousands of small files of equal total size.

Compressing means encoding a file such that the same information is contained in fewer bytes of storage. The advantage for long-term data storage should be obvious. For data transfers, the time spent compressing the data can be balanced against the time saved moving fewer bytes as described in this discussion of data compression and transfer from the US National Center for Supercomputing Applications.

=== Use tar to archive files and directories ===
The primary archiving utility on all Linux and Unix-like systems is the tar command. It will bundle a bunch of files or directories together and generate a single file, called an archive file or tar-file. By convention an archive file has .tar as the file name extension.

When you archive a directory with tar, it will by default include all files and sub-directories contained in it, and sub-sub-directories contained in those, and so on. So

will pack all the contents of directory project1/ into the file project1.tar. The original directory will be unchanged, so this may double the amount of disk space occupied!

You can extract files from the archive using the same command with a different option:

If there is no directory with the original name, it will be created. If a directory of that name exists and contains files of the same names as in the archive file, they will be overwritten.

=== How to compress and uncompress tar files ===
tar can compress an archive file at the same time it creates it. There are a number of compression methods to choose from. We recommend either xz or gzip, which can be used like so:

Typically, --xz will produce a smaller compressed file (a "better compression ratio") but takes longer and uses more RAM while working . --gzip does not typically compress as small, but may be used if you encounter difficulties due to insufficient memory or excessive run time during tar --create.

You can also run tar --create first without compression and then use the commands xz or gzip in a separate step, although there is rarely a reason to do so. Similarly you can run xz -d or gzip -d to decompress an archive file before running tar --extract, but again there is rarely a reason to do so.

=== Common tar options ===
These are the most common options for tar command. There are two synonymous forms for each, a single-letter form prefixed with a single dash, and a whole-word form prefixed with a double dash:
* -c or --create: Create a new archive.
* -f or --file=: Following is the archive file name.
* -x or --extract: Extract files from archive.
* -t or --list: List the contents of an archive file.
* -J or --xz: Compress or uncompress with xz.
* -z or --gzip: Compress or uncompress with gzip.
Single-letter options can be combined with a single dash, so for example

is equivalent to
project1.tar.xz project1}}

There are many more options for tar, and may depend on the version you are using. You can get a complete list of the options available on your system with man tar or tar --help. Note in particular that some older systems might not support --xz compression.