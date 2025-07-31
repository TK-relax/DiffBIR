# test_run.py

# 从我们的模块文件中导入发送邮件的函数
from email_notifier import send_email

# --- 请在这里修改为您的配置信息 ---

# 163邮箱的SMTP服务器地址
MAIL_HOST = "smtp.163.com"

# 您的163邮箱账号
MAIL_USER = "a912206109@163.com"  # 例如: "my_test_sender@163.com"

# 您的163邮箱授权码 (非常重要，不是登录密码！)
MAIL_PASS = "RNeTFnPTtiQSEGqS"  # 例如: "ABCDEFGHIJKLMNOP"

# 发件人邮箱，保持和 MAIL_USER 一致即可
SENDER = "a912206109@163.com"

# 收件人邮箱列表，可以发送给多个收件人
# 这里填写您的QQ邮箱
RECEIVERS = ["1372707774@qq.com"]  # 例如: ["12345678@qq.com"]

# --- 邮件内容配置 ---

# 邮件主题
email_subject = "云端容器训练完成通知"

# 邮件正文
email_content = """
尊敬的用户：

您好！

您在云端容器上执行的训练任务已经成功完成。

请及时查看训练结果。

此邮件为系统自动发送，请勿回复。
"""

# --- 主程序入口 ---
if __name__ == "__main__":
    print("正在尝试发送邮件...")

    # 调用发送邮件的函数
    success = send_email(
        mail_host=MAIL_HOST,
        mail_user=MAIL_USER,
        mail_pass=MAIL_PASS,
        sender=SENDER,
        receivers=RECEIVERS,
        subject=email_subject,
        content=email_content
    )

    if success:
        print("测试邮件已成功发送至指定邮箱，请查收。")
    else:
        print("测试邮件发送失败，请检查配置和网络。")
        print("常见错误原因：")
        print("1. 163邮箱的SMTP服务未开启。")
        print("2. 使用的是邮箱登录密码，而不是授权码。")
        print("3. 邮箱账号或授权码填写错误。")
        print("4. 云端容器网络限制，无法访问外部SMTP服务器的465端口。")