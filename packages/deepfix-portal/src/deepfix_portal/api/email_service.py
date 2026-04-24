"""
Email service for sending verification and password reset emails
"""

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional
from urllib.parse import urlparse

from .config import settings

import aiosmtplib


async def send_verification_email(
    email: str, token: str, name: Optional[str] = None
) -> None:
    """
    Send email verification email to user

    Args:
        email: User's email address
        token: Verification token
        name: User's name (optional)
    """
    config = settings.get_smtp_config()
    frontend_url = settings.get_frontend_url()

    # Check if SMTP is configured
    if not config["username"] or not config["password"]:
        # In development, just log instead of sending
        print(f"[EMAIL] Verification email would be sent to {email}")
        print(f"[EMAIL] Verification link: {frontend_url}/verify-email?token={token}")
        return

    verification_link = f"{frontend_url}/verify-email?token={token}"
    user_name = name or "there"

    # Create email message
    message = MIMEMultipart("alternative")
    message["Subject"] = "Verify your DeepFix account"
    message["From"] = f"{config['from_name']} <{config['from_email']}>"
    message["To"] = email

    # Plain text version
    text_content = f"""
                    Hello {user_name},

                    Welcome to DeepFix! Please verify your email address by clicking the link below:

                    {verification_link}

                    This link will expire in 24 hours.

                    If you didn't create an account with DeepFix, please ignore this email.

                    Best regards,
                    The DeepFix Team
                """

    # HTML version
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
            <h1 style="color: #2563eb; margin-top: 0;">Welcome to DeepFix!</h1>
            <p>Hello {user_name},</p>
            <p>Thank you for signing up! Please verify your email address to complete your account setup.</p>
            <div style="text-align: center; margin: 30px 0;">
                <a href="{verification_link}" style="background-color: #2563eb; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; display: inline-block; font-weight: bold;">Verify Email Address</a>
            </div>
            <p style="font-size: 14px; color: #666;">Or copy and paste this link into your browser:</p>
            <p style="font-size: 12px; color: #999; word-break: break-all;">{verification_link}</p>
            <p style="font-size: 14px; color: #666; margin-top: 30px;">This link will expire in 24 hours.</p>
            <p style="font-size: 14px; color: #666; margin-top: 20px;">If you didn't create an account with DeepFix, please ignore this email.</p>
            <hr style="border: none; border-top: 1px solid #eee; margin: 30px 0;">
            <p style="font-size: 12px; color: #999; margin: 0;">Best regards,<br>The DeepFix Team</p>
        </div>
    </body>
    </html>
    """

    # Attach both versions
    text_part = MIMEText(text_content, "plain")
    html_part = MIMEText(html_content, "html")
    message.attach(text_part)
    message.attach(html_part)

    # Send email
    try:
        await aiosmtplib.send(
            message,
            hostname=config["host"],
            port=config["port"],
            username=config["username"],
            password=config["password"],
            use_tls=config["use_tls"],  # Use STARTTLS for port 587
        )
    except Exception as e:
        # Log error but don't fail the signup process
        print(f"[EMAIL] Failed to send verification email to {email}: {e}")
        raise


async def send_password_reset_email(
    email: str, token: str, name: Optional[str] = None
) -> None:
    """
    Send password reset email to user

    Args:
        email: User's email address
        token: Password reset token
        name: User's name (optional)
    """
    config = settings.get_smtp_config()
    frontend_url = settings.get_frontend_url()

    # Check if SMTP is configured
    if not config["username"] or not config["password"]:
        # In development, just log instead of sending
        print(f"[EMAIL] Password reset email would be sent to {email}")
        print(f"[EMAIL] Reset link: {frontend_url}/reset-password?token={token}")
        return

    reset_link = f"{frontend_url}/reset-password?token={token}"
    user_name = name or "there"

    # Create email message
    message = MIMEMultipart("alternative")
    message["Subject"] = "Reset your DeepFix password"
    message["From"] = f"{config['from_name']} <{config['from_email']}>"
    message["To"] = email

    # Plain text version
    text_content = f"""Hello {user_name},

We received a request to reset your password for your DeepFix account.

Click the link below to reset your password:

{reset_link}

This link will expire in 1 hour.

If you didn't request a password reset, please ignore this email. Your password will remain unchanged.

Best regards,
The DeepFix Team
"""

    # HTML version
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
            <h1 style="color: #2563eb; margin-top: 0;">Reset Your Password</h1>
            <p>Hello {user_name},</p>
            <p>We received a request to reset your password for your DeepFix account.</p>
            <div style="text-align: center; margin: 30px 0;">
                <a href="{reset_link}" style="background-color: #2563eb; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; display: inline-block; font-weight: bold;">Reset Password</a>
            </div>
            <p style="font-size: 14px; color: #666;">Or copy and paste this link into your browser:</p>
            <p style="font-size: 12px; color: #999; word-break: break-all;">{reset_link}</p>
            <p style="font-size: 14px; color: #666; margin-top: 30px;">This link will expire in 1 hour.</p>
            <p style="font-size: 14px; color: #666; margin-top: 20px;">If you didn't request a password reset, please ignore this email. Your password will remain unchanged.</p>
            <hr style="border: none; border-top: 1px solid #eee; margin: 30px 0;">
            <p style="font-size: 12px; color: #999; margin: 0;">Best regards,<br>The DeepFix Team</p>
        </div>
    </body>
    </html>
    """

    # Attach both versions
    text_part = MIMEText(text_content, "plain")
    html_part = MIMEText(html_content, "html")
    message.attach(text_part)
    message.attach(html_part)

    # Send email
    try:
        await aiosmtplib.send(
            message,
            hostname=config["host"],
            port=config["port"],
            username=config["username"],
            password=config["password"],
            use_tls=config["use_tls"],
        )
    except Exception as e:
        print(f"[EMAIL] Failed to send password reset email to {email}: {e}")
        raise
