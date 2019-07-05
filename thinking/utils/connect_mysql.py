import pymysql
import threading
from utils.log_tool_claw import logger
import json
import time


class MysqlDB:
    def __init__(self, conf):
        self.config = {
            'host': conf['host'],
            'user': conf['user'],
            'password': conf['password'],
            'database': conf['database'],
            'charset': 'utf8mb4',  # 支持1-4个字节字符
            'cursorclass': pymysql.cursors.DictCursor
        }
        self.conn = pymysql.connect(**self.config)
        self.cursor = self.conn.cursor()
        self.lock = threading.Lock()
        # Log.sql_log.(error|info|debug)

    def __del__(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def exec_sql(self, strsql):
        """
        write data, such as insert, delete, update
        :param strsql: string sql
        :return: affected rows number
        return 0 when errors
        """
        try:
            self.lock.acquire()
            self.conn.ping(True)
            res = self.cursor.execute(strsql)
            if strsql.strip().lower().startswith("select"):
                res = self.cursor.fetchall()
            self.conn.commit()
            return res
        except Exception as ex:
            logger.error("exec sql error:")
            logger.error(strsql, exc_info=True)
            return 0
        finally:
            self.lock.release()


# def mysql_create():
#     from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, ForeignKey
#     from sqlalchemy import select, and_, or_, not_, text, bindparam
#     # 创建数据库连接
#     engine = create_engine("mysql+mysqlconnector://root:123@127.0.0.1:3306/test?charset=utf8", encoding="utf-8",
#                            echo=True)
#     # 获取元数据
#     metadata = MetaData()
#     # 定义表
#     user = Table('user', metadata,
#                  Column('id', Integer, primary_key=True),
#                  Column('name', String(20)),
#                  Column('fullname', String(40)),
#                  )
#
#     address = Table('address', metadata,
#                     Column('id', Integer, primary_key=True),
#                     Column('user_id', None, ForeignKey('user.id')),
#                     Column('email', String(60), nullable=False)
#                     )
#     # 创建数据表，如果数据表存在，则忽视
#     metadata.create_all(engine)
#     # 获取数据库连接
#     conn = engine.connect()
#
#     i = user.insert()  # 使用查询
#     u = dict(name='jack', fullname='jack Jone')
#     r = conn.execute(i, **u)  # 执行查询，第一个为查询对象，第二个参数为一个插入数据字典，如果插入的是多个对象，就把对象字典放在列表里面
#     print(r.inserted_primary_key)  # 返回插入行 主键 id
#     i = address.insert()
#     addresses = None
#     r = conn.execute(i, addresses)  # 插入多条记录
#     print(r.rowcount)  # 返回影响的行数
#     i = user.insert().values(name='tom', fullname='tom Jim')
#     print(i.compile())
#     print(i.compile().params)
#     r = conn.execute(i)
#     print(r.rowcount)
#     print(select([user]))  # 查询 user表
#     print(user.c)  # 表 user 的字段column对象
#     s = select([user.c.name, user.c.fullname])
#     r = conn.execute(s)
#     print(r.rowcount)  # 影响的行数
#     print(r.fetchall())
#     print(r.closed)  # 只要 r.fetchall() 之后，就会自动关闭 ResultProxy 对象
#     print(select([user.c.name, address.c.user_id]).where(user.c.id == address.c.user_id))  # 使用了字段和字段比较的条件
#
#     se_sql = [(user.c.fullname + ", " + address.c.email).label('title')]
#     wh_sql = and_(
#         user.c.id == address.c.user_id,
#         user.c.name.between('m', 'z'),
#         or_(
#             address.c.email.like('%@aol.com'),
#             address.c.email.like('%@msn.com')
#         )
#     )
#     print(wh_sql)
#     s = select(se_sql).where(wh_sql)
#     print(s)
#     r = conn.execute(s)
#     r.fetchall()
#     text_sql = "SELECT id, name, fullname FROM user WHERE id=:id"  # 原始sql语句，参数用（ ：value）表示
#     s = text(text_sql)
#     print(s)
#     conn.execute(s, id=3).fetchall()  # id=3 传递：id参数
#     print(user.join(address))
#     print(user.join(address, address.c.user_id == user.c.id))  # 手动指定 on 条件
#
#     s = select([user.c.name, address.c.email]).select_from(
#         user.join(address, user.c.id == address.c.user_id))  # 被jion的sql语句需要用 select_from方法配合
#     print(s)
#     print(conn.execute(s).fetchall())
#     s = select([user.c.name]).order_by(user.c.name)  # order_by
#     print(s)
#     s = select([user]).order_by(user.c.name.desc())
#     print(s)
#     s = select([user]).group_by(user.c.name)  # group_by
#     print(s)
#     s = select([user]).order_by(user.c.name.desc()).limit(1).offset(3)  # limit(1).offset(3)
#     print(s)
#     # 更新 update
#     s = user.update()
#     print(s)
#     s = user.update().values(fullname=user.c.name)  # values 指定了更新的字段
#     print(s)
#     s = user.update().where(user.c.name == 'jack').values(name='ed')  # where 进行选择过滤
#     print(s)
#     r = conn.execute(s)
#     print(r.rowcount)  # 影响行数
#     s = user.update().where(user.c.name == bindparam('oldname')).values(
#         name=bindparam('newname'))  # oldname 与下面的传入的从拿书进行绑定，newname也一样
#     print(s)
#     u = [{'oldname': 'hello', 'newname': 'edd'}, {'oldname': 'ed', 'newname': 'mary'},
#          {'oldname': 'tom', 'newname': 'jake'}]
#     r = conn.execute(s, u)
#     print(r.rowcount)
#     # 删除 delete
#     r = conn.execute(address.delete())  # 清空表
#     print(r.rowcount)
#     r = conn.execute(user.delete().where(user.c.name > 'm'))  # 删除记录
#     print(r.rowcount)
#

if __name__ == '__main__':
    try:
        config = {
            'host': "127.0.0.1",
            'user': "root",
            'password': "root",
            'database': "dcos_cmdb",
            'charset': 'utf8mb4',  # 支持1-4个字节字符
            'cursorclass': pymysql.cursors.DictCursor
        }
        mysql = MysqlDB(config)
        service_count_sql = """SELECT COUNT(*) as cot FROM `t_cmdb_server_basic_info`"""
        service_count_phy_sql = """SELECT COUNT(*) as cot FROM `t_cmdb_server_basic_info` WHERE svr_type ='0';"""
        rack_countrack_count_sql = """SELECT COUNT(*) as cot FROM `t_cmdb_server_basic_info` GROUP BY svr_rack_name;"""

        service_count = mysql.exec_sql(service_count_sql)
        service_count_phy = mysql.exec_sql(service_count_phy_sql)
        rack_countrack_count = mysql.exec_sql(rack_countrack_count_sql)
        data = {
            'service_count': service_count[0].get('cot', 0) if service_count else 3500,
            'service_count_phy': service_count_phy[0].get('cot', 0) if service_count_phy else 1000,
            'rack_countrack_count': len(rack_countrack_count) if rack_countrack_count else 50,
            'tenant_count': 24,
        }
    except:
        data = {
            'service_count': 3500,
            'service_count_phy': 1000,
            'rack_countrack_count': 50,
            'tenant_count': 24,
        }
        logger.error("unknown error", exc_info=True)

    try:
        with open('./cmdb_count.json', 'w') as fp:
            fp.write(json.dumps(data))
        logger.info("wrire data success")
    except:
        logger.error("wrire data fail", exc_info=True)
