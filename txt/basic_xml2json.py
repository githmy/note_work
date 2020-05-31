import json
import xmltodict

b = """<?xml version="1.0" encoding="utf-8"?>
        <user_info>
	        <id>12</id>
	        <name>Tom</name>
	        <age>12</age>
	        <height>160</height>
	        <score>100</score>
	        <variance>12</variance>
        </user_info>
    """


def xml_to_json(xml_str):
    # parse是的xml解析器
    xml_parse = xmltodict.parse(xml_str)
    # json库dumps()是将dict转化成json格式,loads()是将json转化成dict格式。
    # dumps()方法的ident=1,格式化json
    json_str = json.dumps(xml_parse, indent=1)
    return json_str


a = {
    "user_info": {
        "id": 12,
        "name": "Tom",
        "age": 12,
        "height": 160,
        "score": 100,
        "variance": 12
    }
}


# json转xml函数
def json_to_xml(json_str):
    # xmltodict库的unparse()json转xml
    # 参数pretty 是格式化xml
    xml_str = xmltodict.unparse(json_str, pretty=1)
    return xml_str


if __name__ == '__main__':
    print("---------------------------分割线----------------------------------")
    print(xml_to_json(b))
    print("---------------------------分割线----------------------------------")
    print(json_to_xml(a))
    print("---------------------------分割线----------------------------------")
