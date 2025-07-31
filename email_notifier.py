# email_notifier.py (已修正版本)

import smtplib
from email.mime.text import MIMEText
from email.header import Header


def send_email(mail_host, mail_user, mail_pass, sender, receivers, subject, content):
    """
    发送邮件的模块函数

    :param mail_host: str, 邮箱服务器地址，例如 'smtp.163.com'
    :param mail_user: str, 发件人邮箱账号，例如 'my_user@163.com'
    :param mail_pass: str, 发件人邮箱授权码（注意：不是登录密码！）
    :param sender: str, 发件人邮箱地址，同 mail_user
    :param receivers: list, 收件人邮箱地址列表，例如 ['user1@qq.com', 'user2@163.com']
    :param subject: str, 邮件主题
    :param content: str, 邮件正文内容
    :return: bool, True 表示发送成功, False 表示发送失败
    """

    message = MIMEText(content, 'plain', 'utf-8')

    # --- 【修改部分】 ---
    # 根据RFC 5322标准，From和To字段应该是标准的邮箱地址格式，不应使用Header编码。
    # 只有当包含非ASCII字符（如中文昵称）时才需要特殊格式化，但直接的邮箱地址不需要。
    # 旧代码: message['From'] = Header(sender, 'utf-8')
    # 新代码: 直接使用字符串，符合邮件标准
    message['From'] = sender

    # 旧代码: message['To'] = Header(",".join(receivers), 'utf-8')
    # 新代码: 直接使用逗号分隔的字符串
    message['To'] = ",".join(receivers)

    # Subject 字段保持不变，因为它经常需要编码来支持中文字符
    message['Subject'] = Header(subject, 'utf-8')
    # --- 【修改结束】 ---

    try:
        # 使用 SMTP_SSL 连接到 SMTP 服务器，SSL加密端口为 465
        smtp_obj = smtplib.SMTP_SSL(mail_host, 465)

        # 登录到邮箱服务器
        smtp_obj.login(mail_user, mail_pass)

        # 发送邮件
        smtp_obj.sendmail(sender, receivers, message.as_string())

        # 关闭连接
        smtp_obj.quit()

        print("邮件发送成功")
        return True

    except smtplib.SMTPException as e:
        print(f"邮件发送失败，错误信息: {e}")
        return False
    except Exception as e:
        print(f"发生未知错误: {e}")
        return False