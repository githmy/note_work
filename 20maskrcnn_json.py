import pandas as pd
import numpy as np
import re
import copy
import json
import os


def main():
    baspth = os.path.join("..", "data", "paper")
    o2_json = json.load(open(os.path.join(baspth, "o2.json"), encoding="utf8"))
    newo2_json = {}
    for key, val in o2_json["_via_img_metadata"].items():
        if len(val['regions']) > 0:
            newo2_json[val['filename']] = val
    print(len(newo2_json))
    my_json = json.load(open(os.path.join(baspth, "my.json"), encoding="utf8"))
    for key, val in my_json["_via_img_metadata"].items():
        for keyf in newo2_json:
            if key.startswith(keyf):
                my_json["_via_img_metadata"][key] = newo2_json[keyf]
    print(len(my_json["_via_img_metadata"]))
    nonelistk = []
    nonelistf = []
    for key, val in my_json["_via_img_metadata"].items():
        if 0 == len(val['regions']):
            nonelistk.append(key)
            nonelistf.append(val['filename'])
    my_json["_via_img_metadata"] = {key: val for key, val in my_json["_via_img_metadata"].items() if
                                    key not in nonelistk}
    print(len(my_json["_via_img_metadata"]))
    parafile = os.path.join(baspth, "res.json")
    uselessfile = os.path.join(baspth, "useless.json")
    json.dump(my_json, open(parafile, mode='w', encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(nonelistf, open(uselessfile, mode='w', encoding="utf-8"), ensure_ascii=False, indent=2)


def findnotin():
    baspth = os.path.join("..", "data", "paper")
    res_json = json.load(open(os.path.join(baspth, "res.json"), encoding="utf8"))
    print(res_json["_via_img_metadata"])
    # listf = [val['filename'] for key, val in res_json["_via_img_metadata"].items()]
    listf = [val['filename'] for val in res_json["_via_img_metadata"].values()]
    dirlist = os.listdir(os.path.join("E:\\", "project", "mark_tool", "试卷"))
    reslist = [file for file in dirlist if file not in listf]
    print(reslist)


def get_ramdom():
    import random
    batp = os.path.join("c:\\", "project", "data", "maskpaper")
    dirlist = os.listdir(os.path.join(batp, "train"))
    slice = random.sample(dirlist, 43)
    for i1 in slice:
        os.system(" ".join(["move", os.path.join(batp, "train", i1), os.path.join(batp, "val")]))
    print(slice)


def interpoints(arrayin):
    arrayout = []
    insernum = 20
    itarry = np.linspace(0, insernum - 1, insernum)
    leth = len(arrayin) - 1
    for oninsert in range(leth):
        avenum = (arrayin[oninsert + 1] - arrayin[oninsert]) / insernum
        arrayout += [arrayin[oninsert] + avenum * i1 for i1 in itarry]
    arrayout += [arrayin[-1]]
    return arrayout


def denseold():
    baspth = os.path.join("c:\\", "project", "data", "testpaper")
    train_json = json.load(open(os.path.join(baspth, "train_bak.json"), encoding="utf8"))
    for key, imgval in train_json["_via_img_metadata"].items():
        newrgion = []
        print(key)
        for regon in imgval["regions"]:
            print(regon["region_attributes"]["markr"])
            # if not regon["region_attributes"]["markr"].startswith("hand"):
            if regon["region_attributes"]["markr"].startswith("title"):
                regon["shape_attributes"]["all_points_x"] = interpoints(regon["shape_attributes"]["all_points_x"])
                regon["shape_attributes"]["all_points_y"] = interpoints(regon["shape_attributes"]["all_points_y"])
                newrgion.append(copy.deepcopy(regon))
            # regon["shape_attributes"]["all_points_x"] = interpoints(regon["shape_attributes"]["all_points_x"])
            # regon["shape_attributes"]["all_points_y"] = interpoints(regon["shape_attributes"]["all_points_y"])
        imgval["regions"] = newrgion
    insertfile = os.path.join(baspth, "train-insert.json")
    json.dump(train_json, open(insertfile, mode='w', encoding="utf-8"), ensure_ascii=False, indent=2)


def get_hw():
    baspth = os.path.join("c:\\", "project", "data", "maskpaper")
    train_json = json.load(open(os.path.join(baspth, "train.json"), encoding="utf8"))
    rec_list = []
    for key, imgval in train_json["_via_img_metadata"].items():
        for regon in imgval["regions"]:
            tw = max(regon["shape_attributes"]["all_points_x"]) - min(regon["shape_attributes"]["all_points_x"])
            th = max(regon["shape_attributes"]["all_points_y"]) - min(regon["shape_attributes"]["all_points_y"])
            rec_list.append(th * tw)
            # rec_list.append(th)
            # rec_list.append(tw)
    data = np.array(rec_list)
    data = np.expand_dims(data, -1)
    print(data)

    from sklearn.cluster import KMeans
    cls = KMeans(n_clusters=5, init='k-means++')
    y_hat = cls.fit_predict(data)
    #
    # from sklearn.cluster import DBSCAN
    # eps = 4
    # min_samples = 100
    # model = DBSCAN(eps=eps, min_samples=min_samples)
    # model.fit(data)
    # y_hat = model.labels_
    print(y_hat)
    print(set(y_hat))
    # pdobj = pd.DataFrame(np.concatenate([data, np.expand_dims(y_hat, -1)], -1))
    pdobj = pd.DataFrame(np.concatenate([np.sqrt(data), np.expand_dims(y_hat, -1)], -1))
    aveobj = pdobj.groupby([pdobj[1]]).mean()
    print(aveobj)


if __name__ == "__main__":
    # get_hw()
    # pass
    denseold()
    # get_ramdom()
    # main()
    # findnotin()
