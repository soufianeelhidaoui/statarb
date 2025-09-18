from __future__ import annotations
import os, smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import yaml
from pathlib import Path


def load_email_config(path: str | Path = "config/email.yaml") -> dict:
    """
    Charge la configuration email à partir de config/email.yaml
    et surcharge le mot de passe via la variable d'environnement EMAIL_PASSWORD.
    """
    if not Path(path).exists():
        return {"enabled": False}

    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    env_pwd = os.getenv("EMAIL_PASSWORD")
    if env_pwd:
        cfg["password"] = env_pwd
    else:
        cfg["password"] = cfg.get("password", "")

    return cfg


def send_email(subject: str, html_body: str, cfg: dict | None = None) -> bool:
    """
    Envoie un email en utilisant les paramètres SMTP de config/email.yaml.
    Sécurisé : password via EMAIL_PASSWORD (variable d'environnement).
    """
    cfg = cfg or load_email_config()
    if not cfg.get("enabled", False):
        return False

    host = cfg["smtp_host"]
    port = int(cfg.get("smtp_port", 587))
    use_tls = bool(cfg.get("use_tls", True))
    skip_verify = bool(cfg.get("skip_tls_verify", False))  # optionnelle

    user = cfg.get("username")
    pwd = cfg.get("password")
    sender = cfg.get("sender", user)
    recipients = cfg.get("recipients", [])
    prefix = cfg.get("subject_prefix", "[StatArb]")

    if not user or not pwd or not recipients:
        raise RuntimeError("Email config invalide : username/password/recipients manquants")

    msg = MIMEMultipart("alternative")
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = f"{prefix} {subject}"

    part = MIMEText(html_body, "html")
    msg.attach(part)

    if skip_verify:
        context = ssl._create_unverified_context()
    else:
        context = ssl.create_default_context()

    with smtplib.SMTP(host, port) as server:
        if use_tls:
            server.starttls(context=context)
        server.login(user, pwd)
        server.sendmail(sender, recipients, msg.as_string())

    return True
