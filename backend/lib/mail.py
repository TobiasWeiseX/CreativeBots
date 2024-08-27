from smtplib import *
from email.mime.text import MIMEText


def send_mail(target_mail, subject, sender_mail, msg):

    msg = MIMEText(msg)
    msg['Subject'] = subject
    msg['From'] = sender_mail
    msg['To'] = target_mail

    smtp = SMTP('mailserver', port=10025)
    smtp.sendmail("Creative Bots", [target_mail], msg.as_string())
    smtp.quit()

