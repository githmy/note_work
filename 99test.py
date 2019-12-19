import pandas as pd
import numpy as np


def main():
    import os
    import json
    import time
    rootpath = os.path.join("e:/", "project", "dataTABLE", "TableBank", "TableBank_data", "Detection_data", "Latex")
    filenames = os.path.join(rootpath, "Latex.json")
    jsobj = json.load(open(filenames))
    # imgnum = {}
    # for i1 in jsobj["annotations"]:
    #     if str(i1["image_id"]) in imgnum:
    #         imgnum[str(i1["image_id"])] += 1
    #     else:
    #         imgnum[str(i1["image_id"])] = 1
    # print(sorted(imgnum.items(), key=lambda x: x[1]))
    idd = 134347
    print(jsobj["images"])
    nmaes = [i1["file_name"] for i1 in jsobj["images"] if i1["id"] == idd]
    print(nmaes)
    for i1 in jsobj["annotations"]:
        if i1["image_id"] == idd:
            print(i1)
    print()
    pass


if __name__ == "__main__":
    main()
