import yagmail


def email_info(headstr, contents, addresses=["a1593572007@126.com"]):
    if addresses is not None or len(addresses) != 0:
        # 连接邮箱服务器
        # yag = yagmail.SMTP(user="1078178757@qq.com", password="my126com", host="smtp.qq.com")
        yag = yagmail.SMTP(user="a1593572007@126.com", password="my126com", host="smtp.126.com")
        # 发送邮件
        yag.send(addresses, "主题：{}".format(headstr), contents=contents)
