import urllib
import base64
# import requests
import json
import simplejson
import urllib.request as librequest
import urllib.parse

# 将文件路径放在此处
file_path = 'lim1it.jpg'
image_uri = "data:image/jpg;base64," + str(base64.b64encode(open(file_path, "rb").read()))
url = "https://api.mathpix.com/v3/latex"
# url = "https://0.0.0.0:80/v3/latex"
request_headers = {"app_id": "trial", "app_key": "34f1a4cea0eaca8540c95908b4dc84ab",
            "Content-type": "application/json"}
endata = bytes(json.dumps({'url': image_uri}), "utf-8")
req = librequest.Request(url=url, data=endata, method='POST', headers=request_headers)
with librequest.urlopen(req) as response:
    ori_page = response.read().decode('utf-8')
    print(ori_page)
    the_page0 = simplejson.loads(ori_page)
    print(json.dumps(json.loads(response.text), indent=4, sort_keys=True))


# r = requests.post("https://api.mathpix.com/v3/latex",
#     data=json.dumps({'url': image_uri}),
#     headers={"app_id": "trial", "app_key": "34f1a4cea0eaca8540c95908b4dc84ab",
#             "Content-type": "application/json"})


"""
curl -X POST https://api.mathpix.com/v3/latex \
    -H 'app_id: trial' \
    -H 'app_key: 34f1a4cea0eaca8540c95908b4dc84ab' \
    -H 'Content-Type: application/json' \
    --data '{ "url": "data:image/jpeg;base64,'$(base64 -i limit1.jpg)'" }'

# 返回
{
    "detection_list": [],
    "detection_map": {
        "contains_chart": 0,
        "contains_diagram": 0,
        "contains_geometry": 0,
        "contains_graph": 0,
        "contains_table": 0,
        "is_inverted": 0,
        "is_not_math": 0,
        "is_printed": 0
    },
    "error": "",
    "latex": "\\lim _ { x \\rightarrow 3} ( \\frac { x ^ { 2} + 9} { x - 3} )",
    "latex_confidence": 0.86757309488734,
    "position": {
        "height": 273,
        "top_left_x": 57,
        "top_left_y": 14,
        "width": 605
    }
}
"""

