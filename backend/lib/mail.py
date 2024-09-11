import os
from jinja2 import Environment, FileSystemLoader
from smtplib import *
from email.message import EmailMessage

env = Environment(loader=FileSystemLoader('templates'))

# Credentials
username = os.getenv("EMAIL_ADDR")
password = os.getenv("EMAIL_PWD")
smtp_domain_and_port = os.getenv("EMAIL_SMTP")

assert username
assert password

def send_mail(target_mail, subject, msg, sender_mail=username):
    html = env.get_template('html_mail.twig').render(
        subject=subject,
        msg=msg
    )

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = sender_mail
    msg['To'] = target_mail
    msg.set_content(html, subtype='html')

    domain, port = smtp_domain_and_port.split(":")

    #with SMTP_SSL('smtp.gmx.de', port=465) as smtp:
    with SMTP_SSL(domain, port=int(port)) as smtp:
        smtp.login(username, password)
        smtp.send_message(msg)
