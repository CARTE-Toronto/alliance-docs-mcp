---
title: "DBaaS/en"
url: "https://docs.alliancecan.ca/wiki/DBaaS/en"
category: "General"
last_modified: "2020-12-02T16:37:43Z"
page_id: 15393
display_title: "DBaaS"
---

__TOC__

== Database as a Service (DBaaS) ==
If a VM is not sufficient to run a database load, a managed database can be used instead, the current offering includes MySQL/MariaDB and Postgres on a physical system.
The database systems as well as all databases are being backed up once a day. The backups are archived for 3 months.
To request access, please contact Technical support.

Please provide in your request the client network or IP address you will access the database from.

Type 	Hostname                               	TCP port
mysql	dbaas101.arbutus.cloud.computecanada.ca	3306
pgsql	dbaas101.arbutus.cloud.computecanada.ca	5432

The CA certificate which is used to sign the host certificate for the service, is available for download here.

== PostgreSQL Database ==
Your instance will use a ssl connection to connect to the DBaaS host.
The example below connects to the DBaaS host, as user01 and uses the database dbinstance via a ssl connection.

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

== MariaDB/MySQL Database ==
Your instance will use a ssl connection to connect to the DBaaS host.
The example below connects to the DBaaS host, as user01 and uses the database dbinstance via a ssl connection.

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