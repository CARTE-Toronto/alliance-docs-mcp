---
title: "Database servers/en"
url: "https://docs.alliancecan.ca/wiki/Database_servers/en"
category: "General"
last_modified: "2025-09-15T18:25:06Z"
page_id: 6825
display_title: "Database servers"
---

== Database servers available for researchers ==
The Alliance offers access to MySQL and Postgres database servers for researchers on both Cedar and Graham.

NOTE: As of January 13, 2025, the Graham cluster will be operating at approximately 25% capacity (see  details here) until Nibi is available. No database server will be provided on Graham during the transition to the new system.

 Database servers

Information                                         	Cedar - MySQL                                                                                              	Cedar - Postgres                                                                                                                                                                     	Graham - MySQL
Description                                         	General purpose server for the researcher wanting to set up SQL tables and issue SQL commands against them.	General purpose server for the researcher wanting to set up SQL tables and issue SQL commands against them. Includes a PostGIS extension available for those needing to do geocoding.	General purpose server for the researcher wanting to set up SQL tables and issue SQL commands against them.
Server name                                         	cedar-mysql-vm.int.cedar.computecanada.ca                                                                  	cedar-pgsql-vm.int.cedar.computecanada.ca                                                                                                                                            	cc-gra-dbaas1.sharcnet.ca  IP: 199.241.163.99
Short server name (can be used instead of long name)	cedar-mysql-vm                                                                                             	cedar-pgsql-vm                                                                                                                                                                       	N/A
Latest version                                      	MariaDB version 11.5                                                                                       	PostgreSQL version 16, PostGIS version 3.3 extension                                                                                                                                 	MariaDB version 11.5
Documentation                                       	MariaDB website                                                                                            	Postgres website, PostGIS documentation                                                                                                                                              	MariaDB website

== Cedar MySQL server ==

The Cedar MySQL server offers MariaDB which is compatible with other flavours of MySQL. For information on compatibility, see MariaDB versus MySQL Compatibility.

The MariaDB server runs as a VM called "cedar-mysql-vm" (full name: cedar-mysql-vm.int.cedar.computecanada.ca).
Users who have accounts on the MySQL server are able to connect only through the Cedar head node
(cedar.computecanada.ca), Cedar compute nodes and Cedar Portal.

For security, users cannot make an SSH connection to the database server directly.

=== MySQL account and connection ===

If you need the privileges to create your own database, you will need a MySQL account. To get a MySQL account on the Cedar MySQL server, please send a request to our Technical support with the following information:

* Your Alliance username
* Amount of database space needed for your project

We will then create an MySQL account with the same username as your Alliance username and a 16 digit random string password. The username, password, database server name, and other information required to make a MySQL connection will be stored in a file called .my.cnf in your home directory. This file is confidential. You cannot change its contents but you can read it or delete it. If the file is deleted, you will lose access to your database.

Run the "mysql" client to connect to the MySQL server. An older version of the client may be available without loading a module, but it will not give you access the latest features on the MySQL server. We recommend issuing the following commands to load a more recent version of the client:
 [name@server ~]$ module load mariadb
 [name@server ~]$ mariadb --version

You can use the following commands to test that your MySQL database account is set up correctly:

Do not use the -p or -h options as arguments when running mysql. The required password and server name are taken automatically from your .my.cnf file.

It is acceptable to submit a long-running SQL command from the head node, as most of the CPU usage is taken from the database server side. However, if you run a script which issues SQL commands and uses lots of CPU, then it needs to be submitted as a job to the scheduler. See Running jobs for details.

=== Set up your MySQL database ===

In order to be able to set up MySQL tables and query them, you need to create your own database. To create a database, the name of the database is arbitrary but it must start with "username_"
where "username" is your MySQL username. For example, if your username were "david" the name of the database must start with "david_" and the commands to create a database called "david_db1", for example, would be:

You may create multiple MySQL databases as long as they all start with "username_".
The created databases will automatically be accessible only to you from the Cedar head node, Cedar compute nodes, and Cedar Portal. You will have full privileges to create SQL objects such as tables, views, etc.

=== Work with your MySQL database ===

Suppose you have account "david" and have created a database called "david_db1" and want to use it. Here's how:

Resources for using MariaDB:
* MariaDB Knowledgebase
* MariaDB Training and Tutorials
* MariaDB SQL Statement Structure
* MariaDB Optimization and Indexes

=== Share your MySQL data ===

All MySQL account holders on Cedar can share their own databases. To share a table of your database with other users:

# Run the command mysql to connect to MySQL.
# Run the command USE database;
#* "database" is the name of the database that you would like to share a table from.
# Run the command GRANT priv_type ON mytable TO ′user′@'172.%';
#* "priv_type" is the type of privilege you would like to grant.
#* "mytable" is the name of the table.
#* "user" is the username you would like to share your table with.

==== MySQL sharing example ====

Username "david" would like to share his table, "mytable" from database, "david_db" with username "john" for reading only. Here are commands he needs to run:

== Cedar PostgreSQL server ==

The Cedar Postgres server has Postgres along with the PostGIS extension.

The Cedar PostgreSQL server runs as a VM called "cedar-pgsql-vm" (full name: cedar-pgsql-vm.int.cedar.computecanada.ca).
Users who have accounts on the PostgreSQL server are able to connect through the main Cedar head node
(cedar.computecanada.ca), Cedar compute nodes, and Cedar Portal.

For security, users cannot make an SSH connection to the database server directly.

To get an account and database on the Cedar PostgreSQL server, send a request to our Technical support with the following information:
* Your Alliance username
* Amount of database space needed for your project
* If you need the PostGIS extension for the database

=== PostgreSQL account and connection ===

The PostgreSQL account we create for you will have the same username as your Alliance username. You will be given a database. The name of the database will typically be "_db" where  is your Alliance username. You cannot create a database yourself. If you need more than one database, please send a request to Technical support.

You do not need to supply a password to access your PostgreSQL account. For security reasons your Alliance password must NEVER be required or used in a script. This also means that one user cannot access another user's database directly.

Run the "psql" client to connect to the PostgreSQL server. An older version of the client may be available without loading a module, but it will not give you access to the latest features of PostgreSQL. We recommend loading the following module to use a more recent version of the client:

 [name@server ~]$ module load postgresql
 [name@server ~]$ psql --version

=== Work with your PostgreSQL database ===

Suppose you have account "david" and have been assigned a database called "david_db". Here is an example of how to use it from the Cedar head node:

> -- List names of tables in your database
 david_db> \dt
 david_db> ... Issue SQL commands. See below for more information.
 david_db> -- Quit
 david_db> \q
}}

Resources for using PostgreSQL:
* PostgreSQL tutorials
* PostgreSQL manuals
* PostgreSQL release notes

=== Share your PostgreSQL data ===
You can share your data in your PostgreSQL database with others. This is how:
* The person with whom you want to share access must have a Postgres account on the system. They should apply for one as described above.
* You will have to give the person connect access to your database.
* For each table or view to be shared, you give the person one or more of select, update, insert, and delete access to it.
* You can also revoke access to a table, view, or database.

Here is an example of user 'david' sharing a table with user 'kim':
> -- Give kim connect access to the database
 david_db> grant connect on database david_db to kim;
 david_db> -- Give kim select-only access to a table called mytable
 david_db> grant select on mytable to kim;
 david_db> -- Quit
 david_db> \q
}}

Here is an example of user 'kim' accessing the shared table:
> -- Connect to the database containing the table to be accessed
 kim_db> \c david_db
 david_db> -- Display the rows in the shared table
 david_db> select * from mytable;
 david_db> -- Quit
 david_db> \q
}}

Here is an example of user 'david' revoking access to 'kim':
> -- Revoke kim's select-only access to a table called mytable
 david_db> revoke select on mytable from kim;
 david_db> -- Revoke kim's connect access to the database
 david_db> revoke connect on database david_db from kim;
 david_db> -- Quit
 david_db> \q
}}

== Graham MySQL server ==
The steps for obtaining and using an account on the Graham MySQL server are similar to those described above for Cedar except you will need to load the following module instead:
 [name@server ~]$ module load mysql

== Cloud-based database servers ==
=== Database as a Service (DBaaS) ===
If a VM is not sufficient to run a database load, a managed database can be used instead, the current offering includes MySQL/MariaDB and Postgres on a physical system.
The database systems as well as all databases are being backed up once a day. The backups are archived for 3 months.
To request access, please contact Technical support.

Please provide in your request the client network or IP address you will access the database from.

Type 	Hostname                               	TCP port
mysql	dbaas101.arbutus.cloud.computecanada.ca	3306
pgsql	dbaas101.arbutus.cloud.computecanada.ca	5432

The CA certificate which is used to sign the host certificate for the service, is available for download here.

=== PostgreSQL database ===
Your instance will use an ssl connection to connect to the DBaaS host.
The example below connects to the DBaaS host, as user01 and uses the database dbinstance via an ssl connection.

psql --set "sslmode=require" -h dbaas101.arbutus.cloud.computecanada.ca -U user01 -d dbinstance
Password for user user01:
SSL connection (protocol: TLSv1.3, cipher: TLS_AES_256_GCM_SHA384, bits: 256, compression: off)
dbinstance=> \l dbinstance
                               List of databases
    Name    | Owner  | Encoding |   Collate   |    Ctype    | Access privileges
------------+--------+----------+-------------+-------------+-------------------
 dbinstance | user01 | UTF8     | en_US.UTF-8 | en_US.UTF-8 | user01=CTc/user01
(1 row)

The ssl connection is enforced and plain text connections fail.

=== MariaDB/MySQL database ===
Your instance will use an ssl connection to connect to the DBaaS host.
The example below connects to the DBaaS host, as user01 and uses the database dbinstance via an ssl connection.

mysql --ssl -h dbaas101.arbutus.cloud.computecanada.ca -u user01 -p dbinstance
Enter password:
MariaDB [dbinstance]> show databases;
+--------------------+
| Database           |
+--------------------+
| dbinstance         |
| information_schema |
+--------------------+
2 rows in set (0.001 sec)

If you try to use a plain text connection, your authentication will fail.

mysql -h dbaas101.arbutus.cloud.computecanada.ca -u user01 -p dbinstance
Enter password:
ERROR 1045 (28000): Access denied for user 'user01'@'client.arbutus' (using password: YES)