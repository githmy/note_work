import requests
import json
import os

def get_citycode():
    citycode = {i1["city_name"]: i1["city_code"] for i1 in
                json.load(open(os.path.join("09weather.json"), encoding="utf8"))}
    return citycode


def 天气(city_name):
    # api地址
    url = 'http://t.weather.sojson.com/api/weather/city/'
    # 通过城市的中文获取城市代码
    citycode = get_citycode()
    city_code = citycode[city_name]
    # 网络请求，传入请求api+城市代码
    response = requests.get(url + city_code)
    # 将数据以json形式返回，这个d就是返回的json数据
    d = response.json()
    dd = "城市：" + d["cityInfo"]["parent"] + d["cityInfo"]["city"] + \
         "\n时间：" + d["time"] + d["data"]["forecast"][0]["week"] + \
         "\n温度：" + d["data"]["forecast"][0]["high"] + d["data"]["forecast"][0]["low"] + \
         "\n天气：" + d["data"]["forecast"][0]["type"] + \
         "\n注意：" + d["data"]["forecast"][0]["notice"]
    print(dd)


# In[22]:
if '__main__' == __name__:
    city_name = "潍坊"
    天气(city_name)
