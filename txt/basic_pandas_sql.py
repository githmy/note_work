from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import pandas as pd
import mysql.connector as cnr
# pip install mysql-connector-python
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, ForeignKey

# pip install SQLAlchemy

def read():
    # 创建数据库连接
    connection = cnr.connect(
        user="root",
        password="123",
        host="127.0.0.1",
        database="test",
        port=3306
    )
    data = pd.read_sql(
        "SELECT * from tt_orderdetail GROUP BY OrderDetailID;",
        con=connection
    )
    # 关闭数据库连接
    connection.close()
    print(data)


def write():
    # 创建数据库连接
    connection = create_engine("mysql+mysqlconnector://root:123@127.0.0.1:3306/test?charset=utf8", encoding="utf-8",
                               echo=True)

    data = pd.DataFrame({
        "age": [21, 22, 23],
        "name": ["a", "b", "c"],
    })
    data.to_sql(
        "data",
        con=connection,
        if_exists="append"
    )
    print(data)


def main():
    # read()
    write()


if __name__ == '__main__':
    main()
    print("end")
