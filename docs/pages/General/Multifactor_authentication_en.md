---
title: "Multifactor authentication/en"
url: "https://docs.alliancecan.ca/wiki/Multifactor_authentication/en"
category: "General"
last_modified: "2025-07-16T16:11:06Z"
page_id: 22441
display_title: "Multifactor authentication"
---

`<languages />`{=html}

Multifactor authentication (MFA) allows you to protect your account with more than a password. Once your account is configured to use this feature, you will need to enter your username and password as usual, and then perform a second action (the `<i>`{=html}second factor`</i>`{=html}) to access most of our services.\
You can choose any of these factors for this second authentication step:

- Approve a notification on a smart device through the Duo Mobile application.
- Enter a code generated on demand.
- Push a button on a hardware key (YubiKey).

This feature will be gradually deployed and will not be immediately available for all of our services.

# Recorded webinars {#recorded_webinars}

Two webinars were presented in October 2023. Their recordings are available here:

- [Authentification multifacteur pour la communauté de recherche](https://www.youtube.com/watch?v=ciycOUbchl8&ab_channel=TheAlliance%7CL%E2%80%99Alliance) (French)
- [Multifactor authentication for researchers](https://www.youtube.com/watch?v=qNsUsZ73HP0&ab_channel=TheAlliance%7CL%E2%80%99Alliance) (English)

# Registering factors {#registering_factors}

## Registering multiple factors {#registering_multiple_factors}

When you enable multifactor authentication for your account, we `<b>`{=html}strongly recommend`</b>`{=html} that you configure at least two options for your second factor. For example, you can use a phone and single-use codes; a phone and a hardware key; or two hardware keys. This will ensure that if you lose one factor, you can still use your other one to access your account.

## Use a smartphone or tablet {#use_a_smartphone_or_tablet}

1.  Install the Duo Mobile authentication application from the [Apple Store](https://itunes.apple.com/us/app/duo-mobile/id422663827) or [Google Play](https://play.google.com/store/apps/details?id=com.duosecurity.duomobile). Make sure to get the correct application (see icon below). TOTP applications such as Aegis, Google Authenticator, and Microsoft Authenticator are `<b>`{=html}not`</b>`{=html} compatible with Duo and will not scan the QR code.
2.  Go to the [CCDB](https://ccdb.alliancecan.ca), log in to your account and select `<i>`{=html}My account → [Multifactor authentication management](https://ccdb.alliancecan.ca/multi_factor_authentications)`</i>`{=html}.
3.  Under `<i>`{=html}Register a device`</i>`{=html}, click on `<i>`{=html}Duo Mobile`</i>`{=html}.
4.  Enter a name for your device. Click on `<i>`{=html}Continue`</i>`{=html}. A QR code will be displayed.
5.  In the Duo Mobile application, tap `<i>`{=html}Set up account`</i>`{=html} or the "+" sign.
6.  Tap `<i>`{=html}Use a QR code`</i>`{=html}.
7.  Scan the QR code shown to you in CCDB. `<b>`{=html}Important: Make sure that your mobile device is connected to the internet (over wi-fi or cellular data) while you are scanning the QR code.`</b>`{=html}

<File:Duo-mobile-app-icon.png%7CStep> 1 <File:Duo-mobile-option-en.png%7CStep> 3 <File:Naming-duo-mobile-device-en.png%7CStep> 4 <File:Duo-mobile-add-account.png%7CStep> 5 <File:Duo-mobile-scan-qr-code.png%7CStep> 6 <File:Scanning-CCDB-QR-code.jpg%7CStep> 7

## Use a YubiKey {#use_a_yubikey}

A YubiKey is a hardware token made by the [Yubico](https://www.yubico.com/) company. If you do not have a smartphone or tablet, do not wish to use your phone or tablet for multifactor authentication, or are often in a situation when using your phone or tablet is not possible, then a YubiKey is your best option.

`<b>`{=html}Note that some YubiKey models are not compatible because they don\'t all support the \"Yubico OTP\" function, which is required. We recommend using the YubiKey 5 Series, but older devices you may already have could work, see this [Yubico identification page](https://www.yubico.com/products/identifying-your-yubikey/) for reference.`</b>`{=html}

A YubiKey 5 is the size of a small USB stick and costs between \$67 and \$100. Different models can fit in USB-A, USB-C, or Lightning ports, and some also support near-field communication (NFC) for use with a phone or tablet.

Multiple protocols are supported by YubiKeys. Our clusters use the Yubico One-Time Password (OTP). After you have registered a YubiKey for multifactor authentication, when you log on to one of our clusters you will be prompted for a one-time password (OTP). You respond by touching a button on your YubiKey, which generates a string of characters to complete your authentication. Using a YubiKey does not require any typing on the keyboard: the YubiKey connected to your computer "types" the string when you touch its button.

To register your YubiKey you will need its Public ID, Private ID, and Secret Key. If you have this information, go to the [Multifactor authentication management page](https://ccdb.computecanada.ca/multi_factor_authentications). If you do not have this information, configure your key using the steps below.

### Configuring your YubiKey for Yubico OTP {#configuring_your_yubikey_for_yubico_otp}

1.  Download and install the YubiKey Manager software from the [Yubico website](https://www.yubico.com/support/download/yubikey-manager/).
2.  Insert your YubiKey and launch the YubiKey Manager software.
3.  In the YubiKey Manager software, select `<i>`{=html}Applications`</i>`{=html}, then `<i>`{=html}OTP`</i>`{=html}. (Images below illustrate this and the next few steps.)
4.  Select `<i>`{=html}Configure`</i>`{=html} for either slot 1 or slot 2. Slot 1 corresponds to a short touch (pressing for 1 to 2.5 seconds), while slot 2 is a long touch on the key (pressing for 3 to 5 seconds). Slot 1 is typically pre-registered for Yubico cloud mode. If you are already using this slot for other services, either use slot 2, or click on `<i>`{=html}Swap`</i>`{=html} to transfer the configuration to slot 2 before configuring slot 1.
5.  Select `<i>`{=html}Yubico OTP`</i>`{=html}.
6.  Select `<i>`{=html}Use serial`</i>`{=html}, then generate a private ID and a secret key. `<b>`{=html}Securely save a copy of the data in the Public ID, Private ID, and Secret Key fields before you click on `<i>`{=html}Finish`</i>`{=html}, as you will need the data for the next step.`</b>`{=html}
7.  `<b>`{=html}IMPORTANT: Make sure you clicked on \"Finish\" in the previous step.`</b>`{=html}
8.  Log into the CCDB to register your YubiKey in the `<i>`{=html}[Multifactor authentication management page](https://ccdb.alliancecan.ca/multi_factor_authentications)`</i>`{=html}.

<File:Yubico> Manager OTP.png\|Step 3 <File:Yubico> Manager OTP configuration.png\|Step 4 <File:Select> Yubico OTP.png\|Step 5 <File:Generate> Yubikey IDs.png\|Step 6, Step 7 CCDB Yubikeys.png\|Step 8

You can test your Yubikey setup by pressing the button on it any time while it is inserted into your computer. If set up correctly, it should generate a code at your prompt or cursor.

# Using your second factor {#using_your_second_factor}

## When connecting via SSH {#when_connecting_via_ssh}

When you connect to a cluster using SSH, you will be prompted for your second factor after you first supply either your password or your [SSH key](https://docs.alliancecan.ca/SSH_Keys "SSH key"){.wikilink}. This prompt will look like this:

At this point, you can select which phone or tablet you want Duo to send a notification to. If you have multiple devices enrolled, you will be shown a list. You will then get a notification on your device, which you accept to complete the authentication.

If you are using a YubiKey, simply touch the YubiKey when the \"Passcode\" prompt appears. If you are using a backup code or a time-based one-time password that the Duo Mobile application shows, you will have to paste it or type it at the prompt.

### Configuring your SSH client with ControlMaster {#configuring_your_ssh_client_with_controlmaster}

#### Linux and MacOS {#linux_and_macos}

If you use OpenSSH to connect, you can reduce how frequently you are asked for a second factor. To do so, edit your `.ssh/config` to add the lines:

    Host HOSTNAME
        ControlPath ~/.ssh/cm-%r@%h:%p
        ControlMaster auto
        ControlPersist 10m

where you would replace `HOSTNAME` with the host name of the server for which you want this configuration. This setting allows a first SSH session to ask for the first and second factors, but subsequent SSH connections on the same device will reuse the connection of the first session (without asking for authentication), even up to 10 minutes after that first session was disconnected.

Note that the above ControlMaster mechanism (a.k.a. Multiplexing) doesn\'t work with native Windows, in which case [Windows Subsystem for Linux](https://learn.microsoft.com/en-gb/windows/wsl/about) will be required. [See the link below](https://docs.alliancecan.ca/wiki/Configuring_WSL_as_a_ControlMaster_relay_server).

#### Windows

See [Configuring WSL as a ControlMaster relay server](https://docs.alliancecan.ca/Configuring_WSL_as_a_ControlMaster_relay_server "Configuring WSL as a ControlMaster relay server"){.wikilink}.

## When authenticating to our account portal {#when_authenticating_to_our_account_portal}

Once multifactor authentication is enabled on your account, you will be required to use it when connecting to our account portal. After entering your username and password, you will see a prompt similar to this, where you click on the option you want to use.\
(Note: `<i>`{=html}This screen will be updated`</i>`{=html}.)

<File:CCDB> MFA prompt.png

# Configuring common SSH clients {#configuring_common_ssh_clients}

Command line clients will typically support multifactor authentication without additional configuration. This is however often not the case for graphical clients. Below are instructions specific to a few of them.

## FileZilla

FileZilla will ask the password and second factor each time a transfer is initiated because by default, transfers use independent connections which are closed automatically after some idle time.

To avoid entering the password and second factor multiple times, you can limit the number of connections to each site to "1" in "Site Manager" =\> "Transfer Settings tab"; note that you'll then lose the ability to browse the server during transfers.

1.  Launch FileZilla and select "Site Manager"
2.  From the "Site Manager", create a new site (or edit an existing one)
3.  On the "General" tab, specify the following:
    - Protocol: "SFTP -- SSH File Transfer Protocol"
    - Host: \[the cluster login hostname\]
    - Logon Type: "Interactive"
    - User: \[your username\]
4.  On the "Transfer Settings" tab, specify the following:
    - Limit number of simultaneous connections: \[checked\]
    - Maximum number of connections: 1
5.  Select "OK" to save the connection
6.  Test the connection

### Niagara special case {#niagara_special_case}

Connections in FileZilla can only be configured to use either SSH keys or interactive prompts, not both. Since Niagara requires using SSH keys and an MFA prompt, using FileZilla is challenging. We recommend using a different SCP client that has better support for interactive prompt, but one possible way to work around is to:

1.  Attempt to connect with an SSH key. This will fail because of the interactive prompt for the second factor. FileZilla will then remember your key.
2.  Change the login method to interactive and attempt to connect again. You will then receive the 2FA prompt.

## MobaXTerm

Install version 23.1 or later. [Version 23.5](https://web.archive.org/web/20231214123606/mobaxterm.mobatek.net/download-home-edition.html) (on Archive.org) is the latest version for which the following instructions work for most users.

#### Prompt on file transfer {#prompt_on_file_transfer}

When connecting to a remote server, MobaXterm establishes two connections by default: the first for the terminal and the second for the remote file browser. By default, the file browser uses the `<i>`{=html}SFTP protocol`</i>`{=html}, which causes a mandatory second prompt for your second factor of authentication.

This behaviour can be improved by switching the `<i>`{=html}SSH-browser type`</i>`{=html} to \"SCP (enhanced speed)\" or \"SCP (normal speed)\" in the session\'s `<i>`{=html}Advanced SSH settings`</i>`{=html}.

#### Use SSH key instead of password {#use_ssh_key_instead_of_password}

To resolve the following issues (1) allow downloads and (2) use SSH passphrase instead of Digital Research Alliance of Canada password, make the following changes to SSH settings (SSH tab in Settings dialogue):

1.  Uncheck \"GSSAPI Kerberos\"
2.  Uncheck \"Use external Pageant\"
3.  Check \"Use internal SSH agent \"MobAgent\"\"
4.  Use the \"+\" button to select SSH key file.

#### Known issues with MFA {#known_issues_with_mfa}

We noticed that after adoption of MFA, MobaXTerm presents a strange behavior, more or less prevalent depending on the version. Although files can be opened via the terminal, when you try to open, download, or upload files using the navigation bar on the left, operations hang indefinitely.

Basically there are pretty much 3 independent sessions that need to be initiated and authenticated when you use MobaXterm:

1.  to open the ssh terminal
2.  to display the contents of the folder on the left pane
3.  to start the transfer of files

It\'s possible that 1 or 2 hidden MFA-Duo windows (behind other windows) on your computer are waiting for authentication.

In addition, each time you navigate to a different folder on the left pane, another transaction requiring MFA is started. Some versions of MobaXterm handle this better than others.

## PuTTY

Install version 0.72 or later.

## WinSCP

Ensure that you are using [SSH Keys](https://docs.alliancecan.ca/SSH_Keys "SSH Keys"){.wikilink}.

## PyCharm

In order to connect to our clusters with PyCharm, you must setup your [SSH Keys](https://docs.alliancecan.ca/SSH_Keys "SSH Keys"){.wikilink} before connecting.

When you connect to a remote host in PyCharm, enter your username and the host you want to connect to. You will then be asked to enter a \"One time password\" during the authentication process. At this stage, use either your YubiKey or your generated password in Duo, depending on what you have setup in your account.

## Cyberduck

By default, Cyberduck opens a new connection for every file transfer, prompting you for your second factor each time. To change this, go in the application\'s preferences, under `<i>`{=html}Transfers`</i>`{=html}, in the `<i>`{=html}General`</i>`{=html} section, use the drop-down menu beside the `<i>`{=html}Transfer Files`</i>`{=html} item and select `<i>`{=html}Use browser connection`</i>`{=html}.

Then, ensure that the box beside `<i>`{=html}Segmented downloads with multiple connections per file`</i>`{=html} is not checked. It should look like the picture below.

![Cyberduck configuration for multifactor authentication](https://docs.alliancecan.ca/CyberDuck_configuration_for_multifactor_authentication.png "Cyberduck configuration for multifactor authentication"){width="400"}

# Frequently asked questions {#frequently_asked_questions}

## Can I use Authy/Google authenticator/Microsoft Authenticator ? {#can_i_use_authygoogle_authenticatormicrosoft_authenticator}

No. Only Duo Mobile will work.

## I do not have a smartphone or tablet, and I do not want to buy a Yubikey {#i_do_not_have_a_smartphone_or_tablet_and_i_do_not_want_to_buy_a_yubikey}

Unfortunately, that means you will not be able to use our services when multifactor authentication becomes mandatory. A Yubikey hardware token is the cheapest way to enable multifactor authentication on your account, and is expected to be covered by the principal investigator\'s research funding like any other work-related hardware. Mandating multifactor authentication is a requirement from our funding bodies.

## Why can\'t you send me one time passcodes through SMS ? {#why_cant_you_send_me_one_time_passcodes_through_sms}

Sending SMS costs money which we do not have. Multifactor using SMS is also widely regarded as insecure by most security experts.

## Why can\'t you send me one time passcodes through email ? {#why_cant_you_send_me_one_time_passcodes_through_email}

No, Duo does not support sending one time code through email.

## I have an older Android phone and I cannot download the Duo Mobile application from the Google Play site. Can I still use Duo ? {#i_have_an_older_android_phone_and_i_cannot_download_the_duo_mobile_application_from_the_google_play_site._can_i_still_use_duo}

Yes. However, you have to download the application from the Duo website:

- For Android 8 and 9, the latest compatible version is [DuoMobile-4.33.0.apk](https://dl.duosecurity.com/DuoMobile-4.33.0.apk)
- For Android 10, the latest compatible version is [DuoMobile-4.56.0.apk](https://dl.duosecurity.com/DuoMobile-4.56.0.apk)

For validation, official [SHA-256 checksums are listed here](https://duo.com/docs/checksums#duo-mobile).

For installation instructions, [see this page](https://help.duo.com/s/article/2211?language=en_US).

## I want to disable multifactor authentication. How do I do this? {#i_want_to_disable_multifactor_authentication._how_do_i_do_this}

Multifactor authentication is mandatory. Users cannot disable it. Exceptions can only be granted for automation purposes. If you find that multifactor authentication is annoying, we recommend applying one of the configurations listed above, depending on the SSH client you are using. Our [recorded webinars](https://docs.alliancecan.ca/Multifactor_authentication#Recorded_webinars "recorded webinars"){.wikilink} also contain many tips on how to make MFA less burdensome to use.

## I do not have a smartphone or tablet, or they are too old. Can I still use multifactor authentication? {#i_do_not_have_a_smartphone_or_tablet_or_they_are_too_old._can_i_still_use_multifactor_authentication}

Yes. In this case, you need [to use a YubiKey](https://docs.alliancecan.ca/#Use_a_YubiKey "to use a YubiKey"){.wikilink}.

## I have lost my second factor device. What can I do? {#i_have_lost_my_second_factor_device._what_can_i_do}

- If you have bypass codes, or if you have more than one registered device, use one of these other mechanisms to connect to your account on our [portal](https://ccdb.alliancecan.ca/multi_factor_authentications). If you have a new device, register it then. Finally, delete your lost device. Note that you cannot delete a device if it is the only one registered.
- If you do not have bypass codes and have lost all of your registered devices, copy the following list providing answers to as many questions as you can. Email this information to support@tech.alliancecan.ca.

` What is the primary email address registered in your account?`\
` For how long have you had an active account with us?`\
` What is your research area?`\
` What is your IP address? (to see your IP address, point your browser to this `[`link`](https://whatismyipaddress.com/)`).`\
` Who is the principal investigator sponsoring your account?`\
` Who are your group members?`\
` Whom can we contact to validate your request?`\
` Which clusters do you use the most?`\
` Which software modules do you load most often on the clusters?`\
` When did you run your last job on the clusters?`\
` Provide a few of your latest batch job IDs on the clusters.`\
` Provide ticket topics and ticket IDs from your recent requests for technical support.`

## Which SSH clients can be used when multifactor authentication is configured? {#which_ssh_clients_can_be_used_when_multifactor_authentication_is_configured}

- Most clients that use a command-line interface, such as on Linux and Mac OS.
- [Cyberduck](https://docs.alliancecan.ca/#Cyberduck "Cyberduck"){.wikilink}
- [FileZilla](https://docs.alliancecan.ca/#FileZilla "FileZilla"){.wikilink}
- JuiceSSH on Android
- [MobaXTerm](https://docs.alliancecan.ca/#MobaXTerm "MobaXTerm"){.wikilink}
- [PuTTY](https://docs.alliancecan.ca/#PuTTY "PuTTY"){.wikilink}
- [PyCharm](https://docs.alliancecan.ca/#PyCharm "PyCharm"){.wikilink}
- Termius on iOS
- VSCode
- [WinSCP](https://docs.alliancecan.ca/#WinSCP "WinSCP"){.wikilink}

## I need to have automated SSH connections to the clusters through my account. Can I use multifactor authentication ? {#i_need_to_have_automated_ssh_connections_to_the_clusters_through_my_account._can_i_use_multifactor_authentication}

We are currently deploying a set of login nodes dedicated to automated processes that require unattended SSH connections. More information about this can be found [here](https://docs.alliancecan.ca/Automation_in_the_context_of_multifactor_authentication "here"){.wikilink}.

## Why have I received the message \"Access denied. Duo Security does not provide services in your current location\" ? {#why_have_i_received_the_message_access_denied._duo_security_does_not_provide_services_in_your_current_location}

Duo blocks authentications from users whose IP address originates in a country or a region subject to economic and trade sanctions: [Duo help](https://help.duo.com/s/article/7544?language=en_US).

# Advanced usage {#advanced_usage}

## Configuring your YubiKey for Yubico OTP using the Command Line (`ykman`) {#configuring_your_yubikey_for_yubico_otp_using_the_command_line_ykman}

1.  Install the command line YubiKey Manager software (`ykman`) following instructions for your OS from Yubico\'s [ykman guide](https://docs.yubico.com/software/yubikey/tools/ykman/Install_ykman.html#download-ykman).
2.  Insert your YubiKey and read key information with the command `ykman info`.
3.  Read OTP information with the command `ykman otp info`.
4.  Select the slot you wish to program and use the command `ykman otp yubiotp` to program it.
5.  `<b>`{=html}Securely save a copy of the data in the Public ID, Private ID, and Secret Key fields. You will need the data for the next step.`</b>`{=html}
6.  Log into the CCDB to register your YubiKey in the `<i>`{=html}[Multifactor authentication management page](https://ccdb.alliancecan.ca/multi_factor_authentications)`</i>`{=html}.

:

``` console
[name@yourLaptop]$ ykman otp yubiotp -uGgP vvcccctffclk 2
Using a randomly generated private ID: bc3dd98eaa12
Using a randomly generated secret key: ae012f11bc5a00d3cac00f1d57aa0b12
Upload credential to YubiCloud? [y/N]: y
Upload to YubiCloud initiated successfully.
Program an OTP credential in slot 2? [y/N]: y
Opening upload form in browser: https://upload.yubico.com/proceed/4567ad02-c3a2-1234-a1c3-abe3f4d21c69
```
