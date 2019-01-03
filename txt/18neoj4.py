import json
from neo4j.v1 import GraphDatabase


# 1.install neo4j

# 2.make neo4j serve as server  con/neo4j.conf
# dbms.connectors.default_listen_address=0.0.0.0
# dbms.connectors.default_advertised_address=0.0.0.0

# 3.start,make constraint for unique node by run 'CREATE CONSTRAINT ON (node:Node) ASSERT node.name IS UNIQUE' in neo4j
# 4.build

# driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "123"))
class GraphNeo4j(object):
    def __init__(self):
        self._driver = GraphDatabase.driver("bolt://47.89.243.140:7687", auth=("neo4j", "123"))

    def close(self):
        self._driver.close()

    def driver_add_node(self, name, type):
        with self._driver.session() as session:
            try:
                session.write_transaction(self.add_node, name, type)
            except Exception as e:
                print(name, type, str(e))

    def driver_add_properties(self, name, type, properties_dict, type_list):
        with self._driver.session() as session:
            try:
                session.write_transaction(self.add_properties, name, type, properties_dict, type_list)
            except Exception as e:
                print(name, type, str(e))

    def driver_add_properties_id(self, name, type, properties_dict, type_list):
        with self._driver.session() as session:
            try:
                session.write_transaction(self.add_properties_id, name, type, properties_dict, type_list)
            except Exception as e:
                print(name, type, str(e))

    def driver_add_properties_id_existence(self, name, type, properties_dict, type_list):
        with self._driver.session() as session:
            try:
                session.write_transaction(self.add_properties_id_existence, name, type, properties_dict, type_list)
            except Exception as e:
                print(name, type, str(e))

    def driver_add_relation(self, name1, type1, relation, name2, type2):
        with self._driver.session() as session:
            try:
                session.write_transaction(self.add_relation, name1, type1, relation, name2, type2)
            except Exception as e:
                print(name1, type1, relation, name2, type2, str(e))

    def driver_add_relation_id(self, id1, name1, type1, relation, id2, name2, type2):
        with self._driver.session() as session:
            try:
                session.write_transaction(self.add_relation_id, id1, name1, type1, relation, id2, name2, type2)
            except Exception as e:
                print(name1, type1, relation, name2, type2, str(e))

    def add_node(self, tx, name, type):
        result = tx.run("MATCH (n:" + type + "{name:'" + name + "'}) RETURN n")
        for record in result:
            if record[0]['name'] == name:
                pass
            else:
                tx.run("CREATE ( n:" + type + "{name:'" + name + "'} )")
            return
        request = "CREATE ( n:" + type + "{name:'" + name + "'} )"
        print("add_node")
        print(request)
        tx.run(request)

    def add_node_id(self, tx, name, type, id):
        result = tx.run("MATCH (n:" + type + "{name:'" + name + "', id:'" + id + "'}) RETURN n")
        for record in result:
            if record[0]['id'] == id:
                pass
            else:
                tx.run("CREATE ( n:" + type + "{name:'" + name + "', id:'" + id + "'})")
            return
        request = "CREATE ( n:" + type + "{name:'" + name + "', id:'" + id + "'})"
        print("add_node")
        print(request)
        tx.run(request)

    def add_node_id_existence(self, tx, name, type, id):
        result = tx.run("MATCH (n:" + type + "{name:'" + name + "', id:'" + id + "'}) RETURN n")
        for record in result:
            if record[0]['id'] == id:
                return True
            else:
                return False
        return False

    def add_properties(self, tx, name, type, properties_dict, type_list):
        # 确认节点存在
        self.add_node(tx, name, type)

        properties_str = ""
        type_str = ""
        for key, value in properties_dict.items():
            properties_str += "SET n." + key + "='" + value + "' "
        for i in type_list:
            type_str += ":" + i
        request1 = "MATCH (n:" + type + "{name:'" + name + "'}) " + properties_str
        request2 = "MATCH (n:" + type + "{name:'" + name + "'})  SET n" + type_str
        print("add_properties")
        if len(properties_dict) > 0:
            tx.run(request1)
        if len(type_list) > 0:
            tx.run(request2)

    def add_properties_id(self, tx, name, type, properties_dict, type_list):
        # 确认节点存在
        # 判断属性id是否一致
        self.add_node_id(tx, name, type, properties_dict['id'])

        properties_str = ""
        type_str = ""
        for key, value in properties_dict.items():
            properties_str += "SET n." + key + "='" + value + "' "
        for i in type_list:
            type_str += ":" + i
        request1 = "MATCH (n:" + type + "{name:'" + name + "', id:'" + properties_dict['id'] + "'}) " + properties_str
        request2 = "MATCH (n:" + type + "{name:'" + name + "', id:'" + properties_dict['id'] + "'})  SET n" + type_str
        print("add_properties")
        if len(properties_dict) > 0:
            tx.run(request1)
        if len(type_list) > 0:
            tx.run(request2)

    def add_properties_id_existence(self, tx, name, type, properties_dict, type_list):
        # 确认节点存在
        # 判断属性id是否一致
        # 不存在时直接结束
        existence = self.add_node_id_existence(tx, name, type, properties_dict['id'])
        if existence:
            properties_str = ""
            type_str = ""
            for key, value in properties_dict.items():
                properties_str += "SET n." + key + "='" + value + "' "
            for i in type_list:
                type_str += ":" + i
            request1 = "MATCH (n:" + type + "{name:'" + name + "', id:'" + properties_dict[
                'id'] + "'}) " + properties_str
            request2 = "MATCH (n:" + type + "{name:'" + name + "', id:'" + properties_dict[
                'id'] + "'})  SET n" + type_str
            print("add_properties")
            if len(properties_dict) > 0:
                tx.run(request1)
            if len(type_list) > 0:
                tx.run(request2)
        else:
            return

    def add_relation_id(self, tx, id1, name1, type1, relation, id2, name2, type2):
        # 确认节点存在
        self.add_node_id(tx, name1, type1, id1)
        self.add_node_id(tx, name2, type2, id2)
        request = "MATCH (n:" + type1 + "{name:'" + name1 + "', id:'" + id1 + "'}) , (m:" + type2 + "{name:'" + name2 + "', id:'" + id2 + "'}) MERGE (n)-[r:" + relation + "]->(m)"
        print("add_relation")
        tx.run(request)

    def add_relation(self, tx, name1, type1, relation, name2, type2):
        # 确认节点存在
        self.add_node(tx, name1, type1)
        self.add_node(tx, name2, type2)
        request = "MATCH (n:" + type1 + "{name:'" + name1 + "'}) , (m:" + type2 + "{name:'" + name2 + "'}) MERGE (n)-[r:" + relation + "]->(m)"
        print("add_relation")
        tx.run(request)

    def add_artist_node(self, data, id):
        # json.dumps(data)
        title_node = data['title_node']
        introduction_node = data['introduction_node']
        basic_node = data['basic_node']
        # music_album_node = data['music_album_node']

        self.driver_add_properties_id_existence(title_node, '歌手', {'id': id, '简介': introduction_node}, [])
        for key, value in basic_node.items():
            # properties_dict = {}
            if key == "职业":
                self.driver_add_properties_id_existence(title_node, '歌手', {'id': id}, value)
            elif len(value) > 1:
                value_string = ""
                for j in value:
                    # self.driver_add_relation(title_node, '歌手', key, j, key)
                    value_string = j + "、"
                self.driver_add_properties_id_existence(title_node, '歌手', {'id': id, key: value_string[:-1]}, [])
            else:
                self.driver_add_properties_id_existence(title_node, '歌手', {'id': id, key: value[0]}, [])
                # for i in music_album_node:
                #     self.driver_add_relation(title_node, '歌手', '专辑', i, '专辑')

    def add_album_node(self, data, id):
        # json.loads(data)
        title_node = data['title_node']
        introduction_node = data['introduction_node']
        basic_node = data['basic_node']
        # music_node = data['music_node']

        self.driver_add_properties_id_existence(title_node, '专辑', {'id': id, '简介': introduction_node}, [])
        for key, value in basic_node.items():
            if key == "音乐风格":
                self.driver_add_properties_id_existence(title_node, '专辑', {'id': id}, value)
            elif len(value) > 1:
                value_string = ""
                for j in value:
                    # self.driver_add_relation(title_node, '歌手', key, j, key)
                    value_string = j + "、"

                self.driver_add_properties_id_existence(title_node, '专辑', {'id': id, key: value_string[:-1]}, [])
            else:
                self.driver_add_properties_id_existence(title_node, '专辑', {'id': id, key: value[0]}, [])
                # for i in music_node:
                #     self.driver_add_relation(title_node, '专辑', '歌曲', i, '歌曲')

    def add_music_node(self, data, id):
        # json.loads(data)
        title_node = data['title_node']
        introduction_node = data['introduction_node']
        basic_node = data['basic_node']
        # music_lyric_node = data['music_lyric_node']

        self.driver_add_properties_id_existence(title_node, '歌曲', {'id': id, '简介': introduction_node}, [])
        for key, value in basic_node.items():
            if len(value) > 1:
                value_string = ""
                for j in value:
                    # self.driver_add_relation(title_node, '歌手', key, j, key)
                    value_string = j + "、"
                self.driver_add_properties_id_existence(title_node, '歌曲', {'id': id, key: value_string[:-1]}, [])
            else:
                self.driver_add_properties_id_existence(title_node, '歌曲', {'id': id, key: value[0]}, [])
                # for i in music_lyric_node:
                #     self.driver_add_properties(title_node, '歌曲', {'歌词':i}, [])

                # with driver.session() as session:
                #     lines=open('./triples.txt','r').readlines()
                #     print(len(lines))
                #     pattern=''
                #     for i,line in enumerate(lines):
                #         arrays=line.split('$$')
                #         name1=arrays[0]
                #         relation=arrays[1].replace('：','').replace(':','').replace('　','').replace(' ','').replace('【','').replace('】','')
                #         name2=arrays[2]
                #         print(str(i))
                #         try:
                #             session.write_transaction(add_node, name1, relation,name2)
                #         except Exception as e:
                #             print( name1, relation,name2,str(e))


if __name__ == '__main__':
    graph = insert_to_neo4j.GraphNeo4j()
    graph.driver_add_node("周杰伦","歌手")
    graph.driver_add_node("Jay","专辑")
    graph.driver_add_relation("周杰伦","歌手", "专辑", "Jay","专辑")
