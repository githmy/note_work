# coding=utf-8
import requests
# ! pip install PyExecJS
import execjs
import json


def run_js():
    js = '''function a(r, o) {
            for (var t = 0; t < o.length - 2; t += 3) {
                var a = o.charAt(t + 2);
                a = a >= "a" ? a.charCodeAt(0) - 87 : Number(a),
                a = "+" === o.charAt(t + 1) ? r >>> a : r << a,
                r = "+" === o.charAt(t) ? r + a & 4294967295 : r ^ a
            }
            return r
        }
        function n(r) {
            var o = r.length;
            o > 30 && (r = "" + r.substr(0, 10) + r.substr(Math.floor(o / 2) - 5, 10) + r.substr(-10, 10));
            var t = void 0
              , n = "" + String.fromCharCode(103) + String.fromCharCode(116) + String.fromCharCode(107);
            t = null !== C ? C : (C = window[n] || "") || "";
            for (var e = t.split("."), h = Number(e[0]) || 0, i = Number(e[1]) || 0, d = [], f = 0, g = 0; g < r.length; g++) {
                var m = r.charCodeAt(g);
                128 > m ? d[f++] = m : (2048 > m ? d[f++] = m >> 6 | 192 : (55296 === (64512 & m) && g + 1 < r.length && 56320 === (64512 & r.charCodeAt(g + 1)) ? (m = 65536 + ((1023 & m) << 10) + (1023 & r.charCodeAt(++g)),
                d[f++] = m >> 18 | 240,
                d[f++] = m >> 12 & 63 | 128) : d[f++] = m >> 12 | 224,
                d[f++] = m >> 6 & 63 | 128),
                d[f++] = 63 & m | 128)
            }
            for (var S = h, u = "" + String.fromCharCode(43) + String.fromCharCode(45) + String.fromCharCode(97) + ("" + String.fromCharCode(94) + String.fromCharCode(43) + String.fromCharCode(54)), l = "" + String.fromCharCode(43) + String.fromCharCode(45) + String.fromCharCode(51) + ("" + String.fromCharCode(94) + String.fromCharCode(43) + String.fromCharCode(98)) + ("" + String.fromCharCode(43) + String.fromCharCode(45) + String.fromCharCode(102)), s = 0; s < d.length; s++)
                S += d[s],
                S = a(S, u);
            return S = a(S, l),
            S ^= i,
            0 > S && (S = (2147483647 & S) + 2147483648),
            S %= 1e6,
            S.toString() + "." + (S ^ h)
        }
        var C = null;
        t.exports = n
    }'''
    ctx = execjs.compile(js)
    ctx.call("n", "你好啊")

def fanyi():
    # headers = {
    #     "User-Agent": "Mozilla/5.0 (Linux; Android 5.1.1; Nexus 6 Build/LYZ28E) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.84 Mobile Safari/537.36"}
    headers = {
        "cookie": "BAIDUID=1AAEEB6DD0B96664B6CAC4419C8557B6:FG=1; PSTM=1548236789; BIDUPSID=CFCFF87599A8061F8A522D570353E768; BDUSS=tTemt0YXBXamdFcDV4d2tLNWZnLXRQZWdOfm1rYkRMNmt4YmJrU2R-enBTaWxkSVFBQUFBJCQAAAAAAAAAAAEAAABWn6cUuty~7ERERkdWAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOm9AV3pvQFdUW; REALTIME_TRANS_SWITCH=1; FANYI_WORD_SWITCH=1; HISTORY_SWITCH=1; SOUND_SPD_SWITCH=1; SOUND_PREFER_SWITCH=1; BDSFRCVID=FOLOJeC626EkeVRwLWXx2QWUQbXHxP6TH6aoEtE4TzhdES91UAeyEG0PDM8g0Ku-0cH4ogKK0eOTHkDF_2uxOjjg8UtVJeC6EG0P3J; H_BDCLCKID_SF=tJkD_I_hJKt3qn7I5KToh4Athxob2bbXHDo-LIkK0hOcOR5JhfA-3R-e046fJpLj-JrI-M5ltqvvhb3O3M7ShbK_KUTLJR8qQm7I2UQF5l8-sq0x0bOte-bQypoa0q3TLDOMahkM5h7xOKQoQlPK5JkgMx6MqpQJQeQ-5KQN3KJmhpFujjL-ej5XjNRf-b-XMTT-04ob26rjDnCrQxndXUI8LNDHtfQW5GcNLhR6yPPKVf3a0t5iybkrjRO7ttoy36n4_UbL2nbpVIIl3fQq2fL1Db3RW6vMtg3tsCJJWIboepvoD-oc3MkfQ-jdJJQOBKQB0KnGbUQkeq8CQft205tpexbH55ueJRFj_MK; BDORZ=B490B5EBF6F3CD402E515D22BCDA1598; delPer=0; BDRCVFR[feWj1Vr5u3D]=I67x6TjHwwYf0; PSINO=5; H_PS_PSSID=1449_29422_21127_29135_29237_28519_29099_28830_29220_20719; locale=zh; yjs_js_security_passport=fc26abaeb29713aaf3235a13ba73b96440c111cd_1562052808_js; from_lang_often=%5B%7B%22value%22%3A%22dan%22%2C%22text%22%3A%22%u4E39%u9EA6%u8BED%22%7D%2C%7B%22value%22%3A%22en%22%2C%22text%22%3A%22%u82F1%u8BED%22%7D%2C%7B%22value%22%3A%22zh%22%2C%22text%22%3A%22%u4E2D%u6587%22%7D%5D; to_lang_often=%5B%7B%22value%22%3A%22zh%22%2C%22text%22%3A%22%u4E2D%u6587%22%7D%2C%7B%22value%22%3A%22en%22%2C%22text%22%3A%22%u82F1%u8BED%22%7D%5D",
        "a": "yjs_js_security_passport=a832f7ce2ba0c3bad519091e5085219e5ad6cae4_1562053639_js",
        "b": "yjs_js_security_passport=fc26abaeb29713aaf3235a13ba73b96440c111cd_1562052808_js",
        "User-Agent": "Mozilla/5.0 (Linux; Android 5.1.1; Nexus 6 Build/LYZ28E) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.84 Mobile Safari/537.36",
    }

    post_data = {
        "query": "加工费呢",
        "from": "zh",
        "to": "en",
        "sign": 14750.318127,
        "token": "a53e0ea8e1b36ac6c7e42b04045422fa",
    }

    r = requests.post("https://fanyi.baidu.com/v2transapi", headers=headers, data=post_data)
    dict_ret = json.loads(r.content.decode())
    print(dict_ret)
    ret = dict_ret["trans_result"]["data"][0]["dst"]
    print("result is :", ret)


if __name__ == "__main__":
    # run_js()
    fanyi()
