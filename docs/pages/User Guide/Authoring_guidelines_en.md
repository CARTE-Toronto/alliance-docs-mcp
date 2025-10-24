---
title: "Authoring guidelines/en"
url: "https://docs.alliancecan.ca/wiki/Authoring_guidelines/en"
category: "User Guide"
last_modified: "2025-09-17T14:33:33Z"
page_id: 108
display_title: "Authoring guidelines"
---

`<languages />`{=html}

## Who can contribute to this Wiki? {#who_can_contribute_to_this_wiki}

Anyone with an Alliance account can contribute. Staff members have the primary responsibility to keep the documentation complete and correct, but this is the age of Wikipedia. An ordinary user who spots an obvious problem, like a dead link or a typographical error, is welcome to fix it. Equally so, a user who is willing to write an entire page on some piece of installed software with which they are very familiar, is also welcome to do that. The documentation team will review it once it's posted to see that it meets these guidelines.

No anonymous editing is possible. You must log in with your Alliance credentials before you are allowed to edit, so we can tell who has done what.

## What belongs on this Wiki? {#what_belongs_on_this_wiki}

This Wiki is not the place for information that properly belongs in the purview of the Alliance communications team, which includes any communications intended for the general public, media, or funding agencies. Materials related to training and outreach also don't belong on this technical documentation site. To that end, ask yourself before you publish a page or make a change:

- Is this about what services or clusters are available? If so, has the service or cluster already been announced? If not, consult the the Senior Manager, Communications & Marketing before publishing.
- Status information which changes from day to day \-\-- available, offline, in maintenance, etc.\-\-- belongs on <https://status.alliancecan.ca/>.
- Is this information useful to a user, as opposed to other CC technical staff? If technical staff, then it might belong at <https://wiki.computecanada.ca/staff/> rather than <https://docs.alliancecan.ca/>.
- Does the information have implications for the security of our systems, or security of data on our systems? If so, consult the Director of Cybersecurity before publishing.
- Is the information of interest only to a prospective user, as opposed to an existing account-holder? This is a gray area: A prospective user might want to know technical details about our services and facilities, the same as an account-holder, but if the information is only of interest to a prospective user then it properly belongs on <https://www.alliancecan.ca> rather than <https://docs.alliancecan.ca/>.
- External links may be appropriate, see e.g. \"Getting an Account\".
- Is this about how to use an existing service, cluster, or application? If so, go ahead.

If you still have any doubt, staff members should use the #rsnt-documentation channel in Slack. Non-staff contributors should contact [Technical support](https://docs.alliancecan.ca/Technical_support "Technical support"){.wikilink}.

## Style guidelines {#style_guidelines}

To the extent possible, we encourage contributors to avoid simply uploading a PDF as this is less than ideal. A better approach is to copy over the relevant text from the PDF and add it to the page, with whatever formatting changes may be needed for a Wiki page, including for example the use of internal links that readers may follow.

### Drafts

If you wish to work on a new page in stages, or get feedback before deciding it is complete, you should mark the page as a draft by inserting

at the top of the source.

### Writing style {#writing_style}

The purpose of a style guide is to support writers in preparing technical documentation that makes learning easier. Carefully crafted documentation appeals to the user and delivers a positive image of the writer. There are several style guides in circulation that set standards for computer documentation. Pioneers in this area are the Apple Style Guide and the Microsoft Manual of Style. There are no official writing guidelines for this wiki, but here are some simple and common practices we can readily adopt:

- Design each paragraph around one idea.
- Present the most important information first.
- Address the reader directly.

:   Example: *The user must click on the button* or *One must click on the button* becomes *Click on the button.*

- Use [simple words and phrases](http://www.plainlanguage.gov/howto/wordsuggestions/simplewords.cfm).
- Use the present tense.

:   Example: *Doing this will launch the XYZ application* becomes *This launches the XYZ application.*

- Use the active voice.

:   Example: *The file is edited by the system administrator* becomes *The system administrator edits the file.*

- Stay positive.

:   Example: *Don\'t use the passive voice* becomes *Use the active voice.*

- Use consistent terms.

:   Yes, synonyms make a text less boring, but for a new user or one reading in a second language, interchangeable terms (e.g. \"machine\", \"host\", \"node\", \"server\") may be confusing.

The word \"system\" is used frequently in computing with different meanings (legacy system, new system, cloud system, file system, module system, job scheduling system, GPU system, storage system, *etc.*). It is not always clear to a new user what we are talking about. Whenever possible, please try to use a more precise word (cluster, storage space, scheduler, *etc.*).

#### External resources {#external_resources}

- Online self-guided [Technical Writing courses from Google](https://developers.google.com/tech-writing/overview).
- [Documentation guide from Write the Docs](https://www.writethedocs.org/guide/).

### Layout style {#layout_style}

When in doubt, imitate the masters. Look at an existing page you like and follow the style. If there isn't one at [docs.alliancecan.ca/](https://docs.alliancecan.ca/), look for one at [ Wikipedia](https://docs.alliancecan.ca/wikipedia: " Wikipedia"){.wikilink}.

- Separate graphic design from content as much as possible. Don't use extra line breaks to adjust vertical spacing. Don't indent paragraphs with tabs or spaces or add extra spaces after a sentence. If we want to make any such style adjustments we will make them universally using stylesheets and templates.
- Screenshots are good, especially in how-tos and tutorials. But full-sized screenshots interrupt the structure and flow of the text if they're placed in-line. Let them float to the right-hand side. Also, scale the image down. If that makes important information unreadable, maybe a cropped picture is better? Or, remind the reader in the caption that they can \"Click on the image for a larger version.\"
- Leave one blank line at the end of each section before the following header. The translation package uses the blank line and header to determine the boundaries of translation units.
- Links to other pages or sites should have a human-oriented description for display rather than the raw URL.
- Capitalize only the first word and [proper nouns](http://blog.apastyle.org/apastyle/2012/02/do-i-capitalize-this-word.html) in titles and headings. Following [ Wikipedia](https://docs.alliancecan.ca/wikipedia:Wikipedia:Naming_conventions_(capitalization) " Wikipedia"){.wikilink}, we prefer the [APA sentence case](http://blog.apastyle.org/apastyle/2012/03/title-case-and-sentence-case-capitalization-in-apa-style.html) for all titles, including page titles.

### Templates

There are multiple [templates](https://docs.alliancecan.ca/Special:UncategorizedTemplates "templates"){.wikilink} available. Please use them as appropriate. Of particular interest are templates for [Including a command within the wiki](https://docs.alliancecan.ca/Including_a_command_within_the_wiki "Including a command within the wiki"){.wikilink} and for [Including a source code file within the wiki](https://docs.alliancecan.ca/Including_a_source_code_file_within_the_wiki "Including a source code file within the wiki"){.wikilink}.

## Translation

To translate a page, one first writes the content in the original language. Second, the page is marked for translation. Then, a *human* translates the page using organizational tools provided by the wiki extension [Translate](https://www.mediawiki.org/wiki/Extension:Translate). Tutorials for this extension can be found [here](https://www.mediawiki.org/wiki/Help:Extension:Translate). Finally, a second human reviews the translation. If a page has not yet been translated, users can see the page in the original language. If a translation has not yet been reviewed, users can see the non-reviewed translation.

Marking a page for translation will trigger an analysis of the content of the wiki page. The page content will be split by the extension into so-called translation units. Translation units can be a title, a paragraph, an image, etc. These small units can then be translated one by one, ensuring that a modification to a page does not trigger the translation of the whole page. This also allows tracking of what percentage of a page is translated, or outdated.

### Mark a new page for translation {#mark_a_new_page_for_translation}

When you have written a page, you should tag it for translation. Here are the steps to do so:

1.  First, if the original language of the page is French, change it on [Special:PageLanguage](https://docs.alliancecan.ca/Special:PageLanguage "Special:PageLanguage"){.wikilink}
2.  Ensure that the content to be translated is enclosed within \<translate\> \</translate\> tags.
3.  Conversely, please enclose code blocks in \</translate\> \<translate\> tags so that they are excluded from translation.
4.  Likewise, try to exclude wiki markup (tables, tags, etc) from translation.
5.  Ensure that the tag \<languages /\> appear at the very top of the page. This will show a box with the list of languages the page is translated into.
6.  Go in "View" mode, and then click on the "Mark this page for translation"
7.  Review the translation units. Check that code blocks and wiki markup are excluded, and all plain text is included.
8.  In the "Priority languages" section, write either "fr" or "en" as the priority language, that is, the language into which it needs to be translated.
9.  Click on "Mark this version for translation"

### Mark changes to a page for translation {#mark_changes_to_a_page_for_translation}

First, try to mark a page for translation only once it is stable. Second, if you do have to make a change to a page that has been translated, make sure you do NOT change the tags of the form \<!\--T:3\--\>. You must never manually edit those tags or copy them. Those are automatically generated.

Once you have done your edits, you can mark the changes to be translated by doing the following :

1.  Ensure that the new text to be translated is enclosed within \<translate\> \</translate\> tags.
2.  Conversely, please enclose code blocks in \</translate\> \<translate\> tags so that they are excluded from translation.
3.  Likewise, try to exclude wiki markup (tables, tags, etc) from translation.
4.  Go in "View" mode. You should see the text "This page has changes since it was last marked for translation." at the top of the page. Click on "marked for translation".
5.  Review the translation units. Check that code blocks and wiki markup are excluded, and all plain text is included.
6.  In the "Priority languages" section, verify that "fr" or "en" appears as the priority language, that is, the language into which it needs to be translated.
7.  Click on "Mark this version for translation"

Note that the \"Page translation\" page includes a checkbox for \"Do not invalidate translations\" in each changed unit. You should only select this option if the change is something like a typo - which shouldn\'t cause the other-language version to need adjustment.

### Code blocks are not translated {#code_blocks_are_not_translated}

Our professional human translator is not a programmer. They cannot distinguish between code and comments in every possible language, so the documentation team has instructed the translator to exclude code blocks from translation.

Putting explanatory comments in code is excellent programming practice which we wish to encourage, but the value of the comments is decreased if the comments aren\'t translated. The documentation team has not found a solution for this that works in every case. Here are some suggestions:

1.  Move the information contained in the comments outside the code block, into the surrounding text (which will then be translated).
2.  Leave an index comment (e.g. \"NOTE 1\", \"NOTE 2\") to connect the external text to the relevant line of code.
3.  If you\'re sufficiently bilingual, and familiar with the translation apparatus, you may translate the code block yourself.

Please *do not* leave example code uncommented, but please *do* remember that comments will not normally be translated, and consider how this will affect the understanding of the user reading the page in translation.

### Translating the sidebar {#translating_the_sidebar}

To add an item that is to be translated in the sidebar, use the following steps:

1.  add the new content to [MediaWiki:Sidebar](https://docs.alliancecan.ca/MediaWiki:Sidebar "MediaWiki:Sidebar"){.wikilink}. Any item which should be translated should be added as either `some-tag` or, if it is a link, `int:some-tag`
2.  add the tags to [MediaWiki:Sidebar-messages](https://docs.alliancecan.ca/MediaWiki:Sidebar-messages "MediaWiki:Sidebar-messages"){.wikilink}
3.  define the content of the tag in English on [MediaWiki:some-tag](https://docs.alliancecan.ca/MediaWiki:some-tag "MediaWiki:some-tag"){.wikilink} (replace `some-tag` by the actual tag)
4.  translate the content of the tag on [this page](https://docs.alliancecan.ca/mediawiki/index.php?title=Special:Translate&language=fr&group=wiki-sidebar&filter=%21translated&action=translate)

## \"Available software\" page {#available_software_page}

Tables on the [Available software](https://docs.alliancecan.ca/Available_software "Available software"){.wikilink} page are automatically generated from module files in CVMFS. In order to add a link to a new page from the \"Documentation\" column of those tables, add an entry to <https://github.com/ComputeCanada/wiki_module_bot/blob/main/module_wiki_page.json>. Please add this change to the definitive copy of the file.

Changes may take six hours to propagate.
