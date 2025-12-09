---
title: "Including a command within the wiki/en"
url: "https://docs.alliancecan.ca/wiki/Including_a_command_within_the_wiki/en"
category: "General"
last_modified: "2019-02-15T17:58:04Z"
page_id: 243
display_title: "Including a command within the wiki"
---

`<languages />`{=html}

To include a command within the wiki, you should use the template. This template detects the **bash** syntax. For example, the code

``` text
```

results in:

## Special characters \"\" and \"\" {#special_characters_and}

Since is a template, the \"=\" and \"\|\" signs are interpreted by the wiki.

To include an equality sign, use the [meta-template ](https://docs.alliancecan.ca/Template:= "meta-template "){.wikilink}. For example, the code:

``` text
$HOME && make && make install}}
```

results in: \$HOME && make && make install}} To include a pipe symbol, use .

## Including a set of commands {#including_a_set_of_commands}

You can use the template to include a set of commands. You may then write each command on a separate line, and prepend the **\|** character in front of each command. For example, the code

``` text
```

results in:

## Modifying the command prompt {#modifying_the_command_prompt}

If you want to modify the command prompt, you may do it by including a *prompt* argument to the template. For example,

``` text
```

results in

In the same way,

``` text
```

results in

## Displaying the result of a command {#displaying_the_result_of_a_command}

You can display the result of a command (and only one) by adding the option `result`. For example,

results in :
