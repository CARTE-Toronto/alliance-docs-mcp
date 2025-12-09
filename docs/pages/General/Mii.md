---
title: "Mii/en"
url: "https://docs.alliancecan.ca/wiki/Mii/en"
category: "General"
last_modified: "2021-10-18T13:53:19Z"
page_id: 17742
display_title: "Mii"
---

A smart search engine for module environments.

Mii works around an existing modules installation, efficiently searching and loading modules on-demand for users.

Once Mii is loaded, modules will be quietly autoloaded for any unambiguous commands. Ambiguous commands will interactively ask for modules to load.

It features:
* Support for Lmod and Environment Modules installations
* bash and zsh shell integration
* Module listing / individual information (via mii list, mii show)
* Searching for exact commands (via mii exact)
* Searching for similar commands (via mii search)
* Optional JSON export format

= Activating Mii =
To enable/load mii:

Once Mii is loaded, it will start making suggestion. When a command is not found, for example:

= Unambiguous commands =
When a command or binary is known and unambiguous, it will be autoloaded:

= Ambiguous commands =
When a command or binary is unknown or ambiguous, it will suggest a selection of potential candidates based on their relevance:

As shown above, we selected one module (#1) out of the selection, and the command was then ran.

= Search with Mii =
You can search for binaries to discover which modules provides it. The results are sorted by relevance:

= Disabling Mii =
To disable mii:

== Re-enabling ==
To re-enable mii: