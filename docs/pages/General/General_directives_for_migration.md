---
title: "General directives for migration/en"
url: "https://docs.alliancecan.ca/wiki/General_directives_for_migration/en"
category: "General"
last_modified: "2023-11-27T16:49:01Z"
page_id: 1608
display_title: "General directives for migration"
---

`<languages/>`{=html}

This page explains issues related to transferring your data between our facilities and our regional partners.

If you are in any doubt about details of the following advice, contact our [technical support](https://docs.alliancecan.ca/technical_support "technical support"){.wikilink} for help.

## What to do before migrating ? {#what_to_do_before_migrating}

Make sure you know whether you are responsible for your own data migration, or whether our staff will be migrating your data. If you are in any doubt, contact our [technical support](https://docs.alliancecan.ca/technical_support "technical support"){.wikilink}.

If you haven\'t used [Globus](https://docs.alliancecan.ca/Globus "Globus"){.wikilink} before, read about it now and verify that it works on the system you are migrating from. Test any other tools you will use (like [tar](http://www.howtogeek.com/248780/how-to-compress-and-extract-files-using-the-tar-command-on-linux/), [gzip](https://www.gnu.org/software/gzip/manual/gzip.html), [zip](https://www.cyberciti.biz/faq/how-to-create-a-zip-file-in-unix/)) on test data to ensure you know how they work before using them on important data.

Do not wait until the last minute to start your migration. Depending on how much data you have and how much load there is on the machines and network, you may be surprised at how long it will take to finish a large transfer. Expect hundreds of gigabytes to take hours to transfer, but give yourself days in case there is a problem. Expect terabytes to take days.

### Clean up {#clean_up}

It is a good practice to look at your files regularly and see what can be deleted, but unfortunately many of us do not have this habit. A major data migration is a good reminder to clean up your files and directories. Moving less data will take less time, and storage space even on new systems is in great demand and should not be wasted.

- If you compile programs and keep source code, delete any intermediate files. One or more of `make clean`, `make realclean`, or `rm *.o` might be appropriate, depending on your [makefile](https://docs.alliancecan.ca/Make "makefile"){.wikilink}.
- If you find any large files named like `core.12345` and you don\'t know that they are, they are probably [core dumps](https://en.wikipedia.org/wiki/Core_dump) and can be deleted.

### Archive and compress {#archive_and_compress}

Most file transfer programs move one file of a reasonable size more efficiently than thousands of small files of equal total size. If you have directories or directory trees containing many small files, use [tar](https://docs.alliancecan.ca/Archiving_and_compressing_files "tar"){.wikilink} to combine (archive) them.

Large files can benefit from compression in some cases, especially text files which can usually be compressed a great deal. Compressing a file `<b>`{=html}only`</b>`{=html} for the purpose of transferring it, and then decompressing it at the end of the transfer will not necessarily save time. It depends on how much the file can be compressed, how long it takes to compress it, and the transfer bandwidth. The calculation is described under `<i>`{=html}Data Compression and transfer discussion`</i>`{=html} in [this document](https://bluewaters.ncsa.illinois.edu/data-transfer-doc) from the US National Center for Supercomputing Applications.

If you decide compression is worthwhile, you can again use [tar](https://docs.alliancecan.ca/Archiving_and_compressing_files "tar"){.wikilink} for this, or [gzip](https://www.gnu.org/software/gzip/manual/gzip.html).

### Avoid duplication {#avoid_duplication}

Try not to move the same data twice. If you are migrating from more than one existing system to one new system and you have data duplicated on the sources, choose one and only move the duplicate data from that one.

Beware of files with duplicate names, but which do not contain duplicate information. Ensure that you will not accidentally overwrite one file with another of the same name.

## What to do during the migration process {#what_to_do_during_the_migration_process}

If it is supported at your source site, use [Globus](https://docs.alliancecan.ca/Globus "Globus"){.wikilink} to set up your file transfer. It is the most user-friendly and efficient tool we know for this task. Globus is designed to recover from network interruptions automatically. We recommend you enable the setting to `<i>`{=html}preserve source file modification times`</i>`{=html} in the `<i>`{=html}Transfer & Timer Options`</i>`{=html}.

If Globus is not supported at your source site, then compressing data and avoiding duplication is even more important. If you are using [scp](https://en.wikipedia.org/wiki/Secure_copy), [sftp](https://en.wikipedia.org/wiki/SSH_File_Transfer_Protocol), or [rsync](https://en.wikipedia.org/wiki/Rsync), then:

- Make a schedule to migrate your data in blocks of a few hundreds of GBs at a time. If the transfer stops for some reason, you will be able to try again starting from the incomplete file, but you will not have to re-transfer files that are already complete. An organized list of files will help here.
- Check regularly to see that the transfer process has not stopped. File size is a good indicator of progress. If no files have changed size for several minutes, then something may have gone wrong. If restarting the transfer does not work, contact our [technical support](https://docs.alliancecan.ca/technical_support "technical support"){.wikilink}.

Be patient. Even with Globus, transferring large volumes of data can be time consuming. Specific transfer speeds will vary, but expect hundreds of gigabytes to take hours and terabytes to take days.

## What to do after migration {#what_to_do_after_migration}

If you did not use Globus, or if you did but did not check `<i>`{=html}verify file integrity`</i>`{=html}, make sure that the data you have transferred are not corrupted. A crude way to do this is to compare file sizes at the source with file sizes at the destination. For greater assurance, you can use [cksum](http://man7.org/linux/man-pages/man1/cksum.1.html) or [md5sum](http://man7.org/linux/man-pages/man1/md5sum.1.html) at each end, and see if the results match. Any files with mismatching sizes or checksums should be transferred again.

## Where and how to get help {#where_and_how_to_get_help}

- To know how to use different archiving and compression utilities, use a Linux command like `man ``<command>`{=html} or `<command>`{=html}` --help`.
- Contact our [technical support](https://docs.alliancecan.ca/technical_support "technical support"){.wikilink}
