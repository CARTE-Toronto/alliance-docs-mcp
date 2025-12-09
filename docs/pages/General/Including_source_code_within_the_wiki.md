---
title: "Including source code within the wiki/en"
url: "https://docs.alliancecan.ca/wiki/Including_source_code_within_the_wiki/en"
category: "General"
last_modified: "2016-05-19T21:57:56Z"
page_id: 232
display_title: "Including source code within the wiki"
---

__NOTOC__

To include source code within the wiki, we are using the extension SyntaxHighlight_GeSHi. You can easily include a code snippet using the tag  .

== Options of the  tag ==
For a complete list of options, please refer to this page.

=== lang option ===
The lang option defines the language used for syntax highlighting. The default language, if this option is omitted, is C++. The complete list of supported languages is available  here.

=== line option ===
The line option displays line numbers.

== Example ==
Here is an example of a C++ code snippet created with the  ...  tag.

#include
#include
#include
#include
using namespace std;

void flushIfBig(ofstream & out, ostringstream & oss, int size, bool force=false) {
	if (oss.tellp() >= size) {
		out << oss.str();
		oss.str(""); //reset buffer
	}
}
int main() {
	int buff_size = 50*1024*1024;

ofstream out ("file.dat");
	ostringstream oss (ostringstream::app);
	oss.precision(5);
	for (int i=0; i<100*buff_size; i++)
	{
		oss << i << endl;
		flushIfBig(out,oss,buff_size);
	}
	flushIfBig(out,oss,buff_size,true);
	out.close();
}