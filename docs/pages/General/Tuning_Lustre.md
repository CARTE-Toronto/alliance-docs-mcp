---
title: "Tuning Lustre/en"
url: "https://docs.alliancecan.ca/wiki/Tuning_Lustre/en"
category: "General"
last_modified: "2022-04-28T13:24:01Z"
page_id: 4105
display_title: "Tuning Lustre"
---

= Lustre Filesystem =

Lustre is a high performance distributed filesystem which allows users to reach high bandwidth for input/output operations. There are however some caveats to consider if one wants to achieve the best performance.
Note that the advice offered here is for advanced users and should be used with caution. Be sure to carry out tests both to verify the scientific validity of your output and to ensure the changes lead to real performance improvements.

== Stripe Count and Stripe Size ==

For each file or directory, it is possible change the stripe size and stripe count parameters. Stripe size is the size of the smallest block of data that is allocated on the filesystem. Stripe count is the number of disks on which the data are spread.

It is possible to get the value of those parameters for a given file or directory using the command

It is also possible to change those parameters for a given directory using the command

For example, if count=8 , then the files will be spread on 8 targets (RAIDs), each MB will be written in a round-robin fashion on up to 8 different servers.

Changing the stripe count will not modify a existing file. To change those parameters, the file must be copied (not moved) to a directory with different parameters or the file needs to be migrated. To create an empty file with a given value of those parameters without changing the parameters of the directory, you may run lfs setstripe on the name of the file to be created. The file will be created as an empty file with the given parameters.

Example of a non-striped directory with a file called "example_file" (lmm_stripe_count is 1 and there is only 1 object for the file)
 $ lfs getstripe striping_example/
 striping_example/
 stripe_count:  1 stripe_size:   1048576 pattern:       raid0 stripe_offset: -1
 striping_example//example_file
 lmm_stripe_count:  1
 lmm_stripe_size:   1048576
 lmm_pattern:       raid0
 lmm_layout_gen:    0
 lmm_stripe_offset: 2
 	obdidx		 objid		 objid		 group
 	     2	       3714477	     0x38adad	   0x300000400

We can change the striping of this directory to use a stripe count of 2 and create a new file.

 $ lfs setstripe -c 2 striping_example
 $ dd if=/dev/urandom of=striping_example/new_file bs=1M count=10
 $ lfs getstripe striping_example/
 striping_example/
 stripe_count:  2 stripe_size:   1048576 pattern:       raid0 stripe_offset: -1
 striping_example//example_file
 lmm_stripe_count:  1
 lmm_stripe_size:   1048576
 lmm_pattern:       raid0
 lmm_layout_gen:    0
 lmm_stripe_offset: 2
 	obdidx		 objid		 objid		 group
 	     2	       3714477	     0x38adad	   0x300000400
 striping_example//new_file
 lmm_stripe_count:  2
 lmm_stripe_size:   1048576
 lmm_pattern:       raid0
 lmm_layout_gen:    0
 lmm_stripe_offset: 3
 	obdidx		 objid		 objid		 group
 	     3	       3714601	     0x38ae29	   0x400000400
 	     0	       3714618	     0x38ae3a	   0x2c0000400

Only the new_file is using the new default of count=2 (lmm_stripe_count) and 2 objects are allocated.

We can restripe the old file using lfs migrate
 $ lfs migrate -c 2 striping_example/example_file
 $ lfs getstripe striping_example/example_file
 striping_example/example_file
 lmm_stripe_count:  2
 lmm_stripe_size:   1048576
 lmm_pattern:       raid0
 lmm_layout_gen:    2
 lmm_stripe_offset: 10
 	obdidx		 objid		 objid		 group
 	    10	       3685344	     0x383be0	   0x500000400
 	    11	       3685328	     0x383bd0	   0x540000400

The file now has a lmm_stripe_count of 2 and 2 objects are allocated

Increasing the stripe count may improve performances, but also makes this file more susceptible to hardware failures.

When a parallel program needs to read a small file (< 1MB), a configuration file for example, it is best to put this file on one disk (stripe count=1), to read it with the master rank, and to send its content to other ranks using a MPI_Broadcast or MPI_Scatter.

When treating large files, it is usually best to use a stripe count as large as the number of MPI ranks. For the stripe size, you will want it to be the same size as the buffer size for the data that is being read or written, by each rank. For example, if each rank reads 1 MB of data at a time, the ideal stripe size will likely be 1 MB. If you don't know what size to use, your best bet is to keep the default value, which has been optimized for large files. Note that the stripe size must always be an integer multiple of 1MB.

In general, you want to reduce the number of open/close operations on the filesystem. It is therefore best to concatenate all data within a single file rather than writing a lot of small files. It will also be best to open the file once at the beginning, and close it once at the end of the program, rather than opening and closing it each time you want to add new data.

== See also ==

* Tools and examples for Archiving and compressing files