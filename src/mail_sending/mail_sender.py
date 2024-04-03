from smtplib import SMTP
import sys
import smtplib
from email.mime.text import MIMEText


EMAIL_USERNAME = 'victor.contrerasordonez@hevs.ch'
EMAIL_PASSWORD = 'vC4n9KAUuz32yd'
EVENT = 'EXTRAAMAS 2024'
ROLE = 'Publicity Chair'
SIGNATURE = 'Victor Hugo Contreras Ordonez <{}>\n{} {}'.format(
    EMAIL_USERNAME, EVENT, ROLE)
AUTHORS_NAMES = ["Victor Hugo"]
AUTHORS_EMAILS = ["victorc365@gmail.com"]

assert len(AUTHORS_NAMES) == len(AUTHORS_EMAILS)
AUTHORS_NAMES_AND_EMAILS = {
    AUTHORS_NAMES[i]: AUTHORS_EMAILS[i] for i in range(len(AUTHORS_NAMES))}

from_addr = EMAIL_USERNAME

subj = "Call for Papers Invitation for {}".format(EVENT)

message_text = "Dear [#AUTHOR_NAME],\n\n"\
    "We are very happy that AAMAS 2024 will be hosting the International Workshop on EXplainable and TRAnsparent AI and Multi-Agent Systems  (EXTRAAMAS 2024) for the sixth time this year!\n"\
    "https://extraamas.ehealth.hevs.ch/\n\n"\
    "We invite you to submit your recent work to the workshop and are looking forward to having vivid and inspiring conversations on recent XAI developments in May!\n"\
    "Deadline for submission: 1 March 2024.\n\n"\
    "Important Dates:\n"\
    "---------------------------\n"\
    "* Deadline for submissions: 01.03.2024\n"\
    "* Notification of acceptance: 25.03.2024\n"\
    "* Registration instructions: 05.04.2024\n"\
    "* Workshop days:  6 - 7.05.2024\n"\
    "* Camera-ready: 15.06.2024\n\n"\
    "Workshop tracks:\n"\
    "------------------------------\n"\
    "* Track 1: XAI in symbolic and subsymbolic AI\n"\
    "* Track 2: XAI in negotiation and conflict resolution\n"\
    "* Track 3: Prompts, Interactive Explainability and Dialogues\n"\
    "* Track 4: XAI in Law and Ethics\n\n\n"\
    "Please check the call for paper here: https://extraamas.ehealth.hevs.ch/docs/CfP_EXTRAAMAS24.pdf\n\n"\
    "All the best!\n"\
    "[#SIGNATURE]"


for name, email in AUTHORS_NAMES_AND_EMAILS.items():
    mailserver = smtplib.SMTP('smtp-mail.outlook.com', 587)
    mailserver.ehlo()
    mailserver.starttls()
    mailserver.ehlo()
    mailserver.login(EMAIL_USERNAME, EMAIL_PASSWORD)
    print('Sending email to "{}" with destination address "{}"'.format(name, email))
    try:
        content = message_text.replace(
            '[#AUTHOR_NAME]', name).replace('[#SIGNATURE]', SIGNATURE)
        msg = MIMEText(content, 'plain')
        msg['Subject'] = subj
        msg['From'] = from_addr
        msg['To'] = email
        try:
            mailserver.sendmail(from_addr, email, msg.as_string())
        finally:
            mailserver.quit()
    except:
        sys.exit('mail failed for name "{}" and email "{}"'.format(name, email))


# with SMTP("outlook.office.com") as smtp:
#    print(smtp.noop())
