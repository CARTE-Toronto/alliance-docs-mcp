---
title: "GBrowse/en"
url: "https://docs.alliancecan.ca/wiki/GBrowse/en"
category: "General"
last_modified: "2025-09-10T18:40:08Z"
page_id: 7066
display_title: "GBrowse"
---

== Introduction ==

GBrowse is a combination of database and interactive web pages for manipulating and displaying annotations on genomes. It requires a web interface to display. GBrowse is installed on Cedar. The web address of the installation is https://gateway.cedar.computecanada.ca.

The Cedar installation differs in some ways from the standard GBrowse setup described at the official website: http://gmod.org/wiki/GBrowse, particularly with regard to authentication and authorization.

==Requesting access to GBrowse==

In order for GBrowse to be able to access your files and directories, our staff will create a shared account for each research group that requests access to GBrowse. While using GBrowse, any member of a research group can read GBrowse config files and input files belonging to any other member of that group. If you wish to use GBrowse, the Principal Investigator (PI) of your group must agree to this change from the usual file security practices. Have the PI write to our technical support indicating that they want a GBrowse account to be created for the group, and that they understand the implications of a shared account.

You must also have a database account on Cedar. If you already have one, please give the name of the database in your email. If you do not already have a database account, please read Database servers carefully and answer the questions given there for setting up a database.

==Setting up GBrowse==

===Config files===

Since GBrowse needs to be able to read config files of all users within a group, place your GBrowse config files in the following directory:

 /project/GROUPID/gbrowse/USERNAME/conf

where GROUPID is your group id and USERNAME is your user name. We will create a symbolic link from ${HOME}/gbrowse-config/ to this directory for your convenience. Files in this directory should be readable by all members of the group, so please do not change the group permission of files in this directory.

===Configuring the database connection===

If you use MySQL, you need the following in your GBrowse config files:

 [username_example_genome:database]
 db_adaptor    =     Bio::DB::SeqFeature::Store
 db_args       =    -adaptor DBI::mysql
 -dsn DATABASE;mysql_read_default_file=/home/SHARED/.my.cnf

where DATABASE is the name of your database and SHARED is the shared account. The .my.cnf file is a text file that is created by our staff. It contains information required for the shared account to make a connection to MySQL.

If you decide to use Postgres, you need the following in your GBrowse config files:

 [username_example_genome:database]
 db_adaptor    = Bio::DB::SeqFeature::Store
 db_args       =  -adaptor DBI::Pg
 -dsn          =  dbi:Pg:dbname=DATABASE

where DATABASE is the name of your database.

==Using GBrowse==

===Input files===

GBrowse is able to read .bam files directly. You do not need to upload them to the database in order to display them. If you want GBrowse to read these .bam files directly:
* Files need to be copied to your /project directory and they should be readable by the group.
* The directory that contains the .bam files must have the setgid and group-execute bits turned on; that is, the output of ls –l must show a small "s" in the group-execute field (not a large "S").
* Make sure that the .bam file's group ownership is set to your group and not to your username. For example, jsmith:jsmith is wrong, jsmith:def-kjones is right.
* Edit your config file to specify the path to the .bam file. Here is an example:

 [example_bam:database]
 db_adaptor        = Bio::DB::Sam
 db_args           = -bam /project/GROUPID/USERNAME/gbrowse_bam_files/example_file.bam
 search options    = default

===Uploading files to the database===

This can be done using BioPerl. Here are commands that need to be run.

 module load bioperl/1.7.1
 bp_seqfeature_load.pl -c –d DATABASE:mysql_read_default_file=/home/USERNAME/.my.cnf \
    example_genomic_sequence.fa header_file

In this example DATABASE is the name of your database and example_genomic_sequence.fa is the FASTA file containing the entire genome that you want to visualize with GBrowse. header_file contains details about the length of the chromosomes. Here is an example of a header file:

##sequence-region I 1 15072434
##sequence-region II 1 15279421
##sequence-region III 1 13783801
##sequence-region IV 1 17493829
##sequence-region V 1 20924180
##sequence-region X 1 17718942
##sequence-region MtDNA 1 13794

We remind you that the above commands should be run via the job scheduler. Do not run these on the head node!

Once you uploaded your data to your database, you need to grant view access to the SHARED account so that GBrowse is able to access your database for reading. Please see How to share your MySQL data.