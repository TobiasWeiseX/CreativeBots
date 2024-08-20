

from smtplib import *
from email.mime.text import MIMEText


"""
import threading
import smtpd
import asyncore
import smtplib

server = smtpd.SMTPServer(('localhost', 1025), None)

loop_thread = threading.Thread(target=asyncore.loop, name="Asyncore Loop")
# If you want to make the thread a daemon
# loop_thread.daemon = True
loop_thread.start()
"""

import asyncio
from aiosmtpd.controller import Controller

class CustomHandler:

    async def handle_DATA(self, server, session, envelope):
        peer = session.peer
        mail_from = envelope.mail_from
        rcpt_tos = envelope.rcpt_tos
        data = envelope.content         # type: bytes
        # Process message data...


        #if error_occurred:
        #    return '500 Could not process your message'

        return '250 OK'



def send_mail2(target_mail, subject, sender_mail, msg):

    #handler = CustomHandler()
    #controller = Controller(handler, hostname='127.0.0.1', port=1025)
    # Run the event loop in a separate thread.
    #controller.start()


    msg = MIMEText(msg)
    msg['Subject'] = subject
    msg['From'] = sender_mail
    msg['To'] = target_mail

    smtp = SMTP('mailserver', port=10025)
    smtp.sendmail("Creative Bots", [target_mail], msg.as_string())
    smtp.quit()


    #controller.stop()




import smtplib, dns.resolver


def send_mail3(target_mail, subject, sender_mail, msg):

    msg = MIMEText(msg)
    msg['Subject'] = subject
    msg['From'] = sender_mail
    msg['To'] = target_mail

    #smtp = SMTP('mailserver', port=10025)
    #smtp.sendmail("Creative Bots", [target_mail], msg.as_string())
    #smtp.quit()


    [nick, domain] = target_mail.split("@")


    #domain = 'example.com'
    records = dns.resolver.resolve(domain, 'MX')
    mx_record = records[0].exchange

    server = smtplib.SMTP(mx_record, 25)

    #server.sendmail('your_email@example.com', 'recipient_email@example.com', 'Hello, this is a test email.')
    server.sendmail(sender_mail, target_mail, msg.as_string())


    server.quit()





import sys
import chilkat

def send_mail(target_mail, subject, sender_mail, msg):


    msg = MIMEText(msg)
    msg['Subject'] = subject
    msg['From'] = sender_mail
    msg['To'] = target_mail




    # The mailman object is used for sending and receiving email.
    mailman = chilkat.CkMailMan()

    recipient = target_mail

    # Do a DNS MX lookup for the recipient's mail server.

    smtpHostname = mailman.mxLookup(recipient)
    if (mailman.get_LastMethodSuccess() != True):
        print(mailman.lastErrorText(), flush=True)
        #sys.exit()
        return False

    print(smtpHostname)

    # Set the SMTP server.
    mailman.put_SmtpHost(smtpHostname)

    # Create a new email object
    email = chilkat.CkEmail()

    email.put_Subject(subject)
    email.put_Body(msg.as_string())
    email.put_From(sender_mail)
    email.AddTo("", recipient)

    success = mailman.SendEmail(email)
    if (success != True):
        print(mailman.lastErrorText(), flush=True)
        return False
    else:
        print("Mail Sent!", flush=True)
        return True



