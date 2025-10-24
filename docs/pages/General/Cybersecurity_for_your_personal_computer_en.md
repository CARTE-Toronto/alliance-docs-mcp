---
title: "Cybersecurity for your personal computer/en"
url: "https://docs.alliancecan.ca/wiki/Cybersecurity_for_your_personal_computer/en"
category: "General"
last_modified: "2022-10-12T18:29:02Z"
page_id: 20818
display_title: "Cybersecurity for your personal computer"
---

`<languages />`{=html}

# Best practices {#best_practices}

Don't know how to keep your personal computer secure? Want to review the security level of your computer?\
Here are a few tips to enhance your computer's security. If you'd like to evaluate its security level, you might like to take this [short quiz for a health check on your computer](https://docs.alliancecan.ca/Cybersecurity:_Personal_computer_health_check "short quiz for a health check on your computer"){.wikilink}.

## Security updates {#security_updates}

Enable *Install Update Automatically* to allow timely installation of security updates on your operating system and software.\
For more information, see [Software updates: Why they matter for cybersecurity](https://www.getcybersafe.gc.ca/en/blogs/software-updates-why-they-matter-cyber-security).

## Passwords

Strong passwords are essential to keep your computer and your accounts secure. Refer to [Password hygiene habits](https://docs.alliancecan.ca/Cybersecurity_for_your_personal_computer#Password_hygiene_habits "Password hygiene habits"){.wikilink} for more tips.

## Antivirus

To prevent your computer from malware infection, install an antivirus software and keep it updated.

## Phishing

Pay attention to the hyperlinks contained in emails or search engine results before you click on them. A hyperlink containing a weird domain name is a strong signal of malicious activities.\
For more information, see [Signs of a phishing campaign: How to keep yourself safe](https://www.getcybersafe.gc.ca/en/blogs/signs-phishing-campaign-how-keep-yourself-safe).

## Wi-Fi security {#wi_fi_security}

To protect the Wi-Fi network at your home, set a strong Wi-Fi password and update your router's firmware regularly.\
Avoid using public Wi-Fi as much as possible. If you need to, consider installing a trustworthy VPN solution and enable it when you connect to a public Wi-Fi spot.\
For more information, see [Private networks](https://www.getcybersafe.gc.ca/en/secure-your-connections/private-networks) and [Public Wi-Fi](https://www.getcybersafe.gc.ca/en/secure-your-connections/public-wi-fi).

## Note

The advice above is mainly for individuals wanting to refresh their cybersecurity awareness and improve cyber defense on their personal computers.\
Computers at the workplace are typically managed and protected by the organization's IT services team where different sets of security measures may be applied. You should follow your organization's policy to protect computers at the workplace.\
[Check out our short quiz for a health check on your computer!](https://docs.alliancecan.ca/Cybersecurity:_Personal_computer_health_check "Check out our short quiz for a health check on your computer!"){.wikilink}

# Password hygiene habits {#password_hygiene_habits}

Despite many solutions that protect information and systems, stolen usernames and passwords (credentials) are still the most common way attackers gain unauthorized access. This is frequently the result of weak, guessable passwords and reused credentials that have been exposed.

What do you think is the best way to keep your passwords secure?\
A. Change them frequently\
B. Use special characters and a mix of lowercase and uppercase letters\
C. Create each password long and unique\
Changing passwords frequently without cause can actually degrade security. When forced to change their password frequently, many people choose an easy one to remember based on predictable patterns. Long passwords can be quite secure, especially when they are unique. Adding complexity to a password can help, but length proves to be more important than the actual characters used. The best answer to this question is to create long passwords AND use a different one for each service. Why? Because breaches do happen and some service will eventually mishandle your credentials, which will then get exposed. Just have a look at <https://haveibeenpwned.com/> to see that this has already happened to many. If your password isn\'t unique and is exposed, it can be used to access any system where your credentials are valid. This process called *password stuffing* is usually automated and can happen as quickly as 12 hours after the initial exposure.

**Best password tips**

- Use a password manager
  - Regardless if you choose one that is standalone or integrated into your web browser, open source or commercial product/service, a password manager is essential when it comes to all the other steps below.
- Use a different password for everything: every service, every system;
  - This is quite easy if you're using a password manager.
- Make it long - 15 characters or longer is a good size;
  - Again, easy with a password manager when you allow it to generate the passwords for you. Using passwords with 20 to 32 characters is not a problem since you don't need to remember them anyway.
- Never share it with anyone\... really\... no one\... ever;
  - Your credentials belong to you, they identify you. Sharing them not only compromises your identity but is also usually a violation of the policies of the service or system they are used to access.
- Change them only if there is a reason.
  - If you believe the password may have been compromised, may be reused, or is weak, you should change it. There is no good reason to change passwords based on a specific schedule, which may still be required by some organizations.

If this is not what you've been doing, **don't panic!** You can start making changes today. If you have hundreds of passwords, start with a few of them, do a couple every day at lunch. Every time you make even one set of credentials more secure you're doing yourself a big favour.

# Safe browsing and MFA {#safe_browsing_and_mfa}

We rely on a variety of online resources and accounts to help us in our work and to tackle tasks effectively. How we access these tools and how we behave online can have a significant impact on our personal security and the security of the resources we share.

Taking control of the information we provide to online service providers, limiting the extent to which commercial entities can track our activity, and thinking about how we authenticate to online accounts can all have a security benefit.

We can start where we are and start today. We can choose to share less personal information voluntarily when responding to requests, signing up for services, posting on social media. The less personal information you share about yourself, the harder it is for an attacker to connect those pieces of information and use them to target you.

We can choose to use privacy-enhancing search tools like DuckDuckGo ([duckduckgo.com](https://duckduckgo.com/)), install browser extensions like Privacy Badger ([privacybadger.org](https://privacybadger.org/)), HTTPS Everywhere ([eff.org/https-everywhere](https://www.eff.org/https-everywhere)), uBlock Origin ([ublockorigin.com](https://ublockorigin.com/)). We can limit the use of cookies via browser settings and turn on features that limit the links and tracking tools of social media companies ([mozilla.org/en-US/firefox/facebookcontainer](https://www.mozilla.org/en-US/firefox/facebookcontainer/)).

When authenticating to online accounts, we can use different identities/usernames/emails for different services; separate work and personal accounts; practice good password hygiene (see our password tips above); and enroll in the MFA schemes provided by online service providers.

Doing even some of these things will make it more challenging for attackers to target us and our colleagues in phishing attacks, to engage in credential stuffing or password guessing.

# Linux permissions {#linux_permissions}

Audience: the content below is intended for a technical audience such as users of our supercomputers.

Linux permissions are one layer of protection to safeguard your research. Here are three common mistakes to avoid:

***Mistake 1***: Granting access to a file to the world via the command ''*chmod 777 name_of_file*''.

Make sure you understand [how Linux permissions work](https://docs.alliancecan.ca/Sharing_data#Filesystem_permissions "how Linux permissions work"){.wikilink} and restrict access to your files in our supercomputers to only those who need access to them.

***Mistake 2***: Not using the *sticky bit*, leading to the deletion of your files by someone else.

When dealing with a shared directory where multiple users have read, write and execute permission, the issue of ensuring that an individual cannot delete the files or directories of another can arise. Make sure you are familiar with [the notion of sticky bit](https://docs.alliancecan.ca/Sharing_data#The_Sticky_Bit "the notion of sticky bit"){.wikilink} and use it when appropriate.

***Mistake 3***: Granting access to multiple individuals rather than groups.

[Managing ACLs (Access Control Lists)](https://docs.alliancecan.ca/Sharing_data#Access_control_lists_.28ACLs.29 "Managing ACLs (Access Control Lists)"){.wikilink} can quickly become complex. It is best practice to use groups rather than multiple individual accounts to grant permissions when possible.
