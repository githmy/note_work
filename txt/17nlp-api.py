from baidubaike import *
from hudongbaike import *
from sogoubaike import *
import jieba

def collect_infos(word):
    baidu = BaiduBaike()
    hudong = HudongBaike()
    sogou = SougouBaike()
    merge_infos = list()
    baidu_infos = baidu.info_extract_baidu(word)
    hudong_infos = hudong.info_extract_hudong(word)
    sogou_infos = sogou.info_extract_sogou(word)
    merge_infos += baidu_infos
    merge_infos += hudong_infos
    merge_infos += sogou_infos

    return merge_infos
