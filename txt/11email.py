import yagmail


def test_email():
    # 连接邮箱服务器
    # yag = yagmail.SMTP(user="1078178757@qq.com", password="my126com", host="smtp.qq.com")
    yag = yagmail.SMTP(user="a1593572007@126.com", password="my126com", host="smtp.126.com")

    # 邮箱正文
    contents = ["第一行",
                "<a href='https://www.baidu.com'>for baidu</a>",
                "第3\n行",
                "../../input/note_work/email_test.png",
                "last行",
                ]

    # 发送邮件
    yag.send(["1272736235@qq.com"], "主题：from abc", contents=contents)


if __name__ == "__main__":
    test_email()
