def server():
    # -*- coding:utf-8 -*-
    import socket
    import os

    line = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    line.bind(('192.168.1.102', 5555))
    # 同时连接个数
    line.listen(5)
    print('waiting commd------->')

    while True:
        conn, addr = line.accept()
        msg = conn.recv(1024)
        if msg == 'q':
            break
        print('get commd:', msg)
        result = os.popen(msg).read()
        conn.send('result\n' + result)

    conn.close()
    line.close()


def client():
    import socket

    while True:
        line = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        line.connect(('192.168.1.102', 5555))
        msg = str(input('please input commd:'))
        print(type(msg))
        # line.send(msg, 'utf-8')
        line.send(str(msg).encode(encoding='utf-8'))
        # line.send(msg.encode('utf-8'))
        data = line.recv(1024)
        print(data)
        line.close()


if __name__ == "__main__":
    client()
