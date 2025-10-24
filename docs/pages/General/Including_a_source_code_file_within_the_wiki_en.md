---
title: "Including a source code file within the wiki/en"
url: "https://docs.alliancecan.ca/wiki/Including_a_source_code_file_within_the_wiki/en"
category: "General"
last_modified: "2018-02-14T12:45:20Z"
page_id: 1380
display_title: "Including a source code file within the wiki"
---

\_\_NOTOC\_\_ `<languages />`{=html} As explained on the page [Including source code within the wiki](https://docs.alliancecan.ca/Including_source_code_within_the_wiki "Including source code within the wiki"){.wikilink}, you can include source code within the wiki using the **\<syntaxhighlight\> \</syntaxhighlight\>** tag. If you want to separate the code a bit more from the rest of the text, you can use the template. This template takes as argument the file name (**name** parameter), the language of the file (**lang** parameter) and the content of the file (**contents** parameter). The default language for this template is *bash*.

For example,

``` text
```

results in:

## Special characters: Pipe, equals {#special_characters_pipe_equals}

Certain characters that frequently appear in bash scripts are also meaningful to the Mediawiki template parser.

- If the source code contains a pipe character, `|`, replace it with .
- In some circumstances you may find it necessary to replace the equal sign, `=`, with .

## Displaying line numbers {#displaying_line_numbers}

To display line numbers, you can add the option \"\|lines=yes\". For example,

``` text
```

results in:
