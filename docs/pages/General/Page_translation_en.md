---
title: "Page translation/en"
url: "https://docs.alliancecan.ca/wiki/Page_translation/en"
category: "General"
last_modified: "2022-07-08T22:34:02Z"
page_id: 220
display_title: "Page translation"
---

`<noinclude>`{=html} `<languages />`{=html}

`</noinclude>`{=html}

To translate a page, one first writes the content in the original language. Second, the page is marked for translation. Then, a *human* translates the page using organizational tools provided by the wiki extension [Translate](https://www.mediawiki.org/wiki/Extension:Translate). Tutorials for this extension can be found [here](https://www.mediawiki.org/wiki/Help:Extension:Translate). Finally, a second human reviews the translation. If a page has not yet been translated, users can see the page in the original language. If a translation has not yet been reviewed, users can see the non-reviewed translation.

Marking a page for translation will trigger an analysis of the content of the wiki page. The page content will be split by the extension into so-called translation units. Translation units can be a title, a paragraph, an image, etc. These small units can then be translated one by one, ensuring that a modification to a page does not trigger the translation of the whole page. This also allows tracking of what percentage of a page is translated, or outdated.

## How does one mark a new page for translation ? {#how_does_one_mark_a_new_page_for_translation}

When you have written a page, you should tag it for translation. Here are the steps to do so:

1.  Ensure that the wiki code to be translated is enclosed within \<translate\> \</translate\> tags.
2.  Ensure that the tag \<languages /\> appear at the very top of the page. This will show a box
3.  Go in "View" mode, and then click on the "Mark this page for translation"
4.  Review the translation units.
    1.  Try to ensure that no wiki code (tables, tags, etc.) gets translated. This can be done by breaking the page in multiple \<translate\> \</translate\> sections.
5.  In the "Priority languages" section, write either "fr" or "en" as the priority language, that is, the language into which it needs to be translated.
6.  Click on "Mark this version for translation"

## How does one mark changes to a page for translation ? {#how_does_one_mark_changes_to_a_page_for_translation}

First, try to mark a page for translation only once it is stable. Second, if you do have to make a change to a page that has been translated, make sure you do NOT change the tags of the form \<!\--T:3\--\>. Those are automatically generated.

Once you have done your edits, you can mark the changes to be translated by doing the following :

1.  Ensure that the new text to be translated is enclosed within \<translate\> \</translate\> tags.
2.  Go in "View" mode. You should see the text "This page has changes since it was last marked for translation." at the top of the page. Click on "marked for translation".
3.  Review the translation units.
    1.  Try to ensure that no wiki code (tables, tags, etc.) gets translated. This can be done by breaking the page in multiple \<translate\> \</translate\> sections.
4.  In the "Priority languages" section, write either "fr" or "en" as the priority language, that is, the language into which it needs to be translated.
5.  Click on "Mark this version for translation"

`<noinclude>`{=html} `</noinclude>`{=html}
