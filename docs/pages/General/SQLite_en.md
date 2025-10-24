---
title: "SQLite/en"
url: "https://docs.alliancecan.ca/wiki/SQLite/en"
category: "General"
last_modified: "2020-03-12T19:47:58Z"
page_id: 13014
display_title: "SQLite"
---

`<languages />`{=html}

[SQLite](https://www.sqlite.org) is a database management tool used to build commonly called *pocket databases* because they offer all the features of relational databases without the client-server architecture and with the added advantage of the data residing on a single disk file which can simply be copied to another computer. Software written in a wide variety of common languages can read from and write to the database file with standard [SQL](https://en.wikipedia.org/wiki/SQL) queries using the language\'s API for database interactions.

Like any other database, an SQLlite database should not be used on a shared filesystem such as home, scratch and project. Typically, you should copy your SQLite file to the local scratch `$SLURM_TMPDIR` space at the beginning of a job and you can then use the database without any issues and also enjoy the best possible performance. Note that SQLite is not intended for use with multiple threads or processes writing concurrently to the database; for this you should consider a [ client-server solution](https://docs.alliancecan.ca/Database_servers " client-server solution"){.wikilink}.

## Using SQLite directly {#using_sqlite_directly}

You can access an SQLite database directly using the native client:

If the file `foo.sqlite` does not already exist, SQLite will create it and the client will start in an empty database, otherwise you will be connected to the existing database. You may then execute whichever queries you wish on the database, such as `SELECT * FROM tablename;` to print to the screen the entire contents of the table `tablename`.

## Accessing SQLite from software {#accessing_sqlite_from_software}

The most common way to interact with an SQLite (or other) database is through function calls to open a connection to the database; execute queries that can read, insert or update existing data; and close the connection to the SQLite database so that any changes are flushed to the SQLite file. In the simple example below, we suppose that the database has already been created with a table called `employee` that has two columns: the string `name` and the integer `age`.

`<tabs>`{=html} `<tab name="Python">`{=html}

``` python
#!/usr/bin/env python3

# For Python we can use the module sqlite3, installed in a virtual environment, 
# to access an SQLite database
import sqlite3

age = 34

# Connect to the database...
dbase = sqlite3.connect("foo.sqlite")

dbase.execute("INSERT INTO employee(name,age) VALUES(\"John Smith\"," + str(age) + ");")

# Close the database connection
dbase.close()
```

`</tab>`{=html} `<tab name="R">`{=html}

``` r
# Using R, the first step is to install the RSQLite package in your R environment, 
# after which you can use code like the following to interact with the SQLite database
library(DBI)

age <- 34

# Connect to the database...
dbase <- dbConnect(RSQLite::SQLite(),"foo.sqlite")

# A parameterized query
query <- paste(c("INSERT INTO employee(name,age) VALUES(\"John Smith\",",toString(age),");"),collapse='')
dbExecute(dbase,query)

# Close the database connection
dbDisconnect(dbase)
```

`</tab>`{=html} `<tab name="C++">`{=html}

``` cpp
#include <iostream>
#include <string>
#include <sqlite3.h>

int main(int argc,char** argv)
{
  int age = 34;
  std::string query;
  sqlite3* dbase;

  sqlite3_open("foo.sqlite",&dbase);

  query = "INSERT INTO employee(name,age) VALUES(\"John Smith\"," + std::to_string(age) + ");";
  sqlite3_exec(dbase,query.c_str(),nullptr,nullptr,nullptr);

  sqlite3_close(dbase);

  return 0;
}
```

`</tab>`{=html} `</tabs>`{=html}

## Limitations

As its name suggests, SQLite is easy to use and intended for relatively simple databases which are neither excessively large (hundreds of gigabytes or more) nor too complicated in terms of their [entity-relationship diagram](https://en.wikipedia.org/wiki/Entity%E2%80%93relationship_model). As your SQLite database grows in size and complexity, the performance could start to degrade, in which case the time may have come to consider the use of more [ sophisticated database software which uses a client-server model](https://docs.alliancecan.ca/Database_servers " sophisticated database software which uses a client-server model"){.wikilink}. The SQLite web site includes an excellent page on [Appropriate Uses For SQLite](https://www.sqlite.org/whentouse.html), including a checklist for choosing between SQLite and client-server databases.
