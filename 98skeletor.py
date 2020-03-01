import pandas as pd
import numpy as np
import re
import json
import os

pathf = os.path.join("..", "data", "particles")


def getdata():
    trainpd = pd.read_csv(os.path.join(pathf, "train.csv"))
    print(trainpd.head(1))
    print(trainpd.shape)
    eventpd = pd.read_csv(os.path.join(pathf, "event.csv"))
    print(eventpd.head(1))
    print(eventpd.shape)
    samplepd = pd.read_csv(os.path.join(pathf, "sample.csv"))
    print(samplepd.head(1))
    print(samplepd.shape)
    testpd = pd.read_csv(os.path.join(pathf, "test.csv"))
    print(testpd.head(1))
    print(testpd.shape)
    exit()
    return testpd


def main():
    res = getdata()
    print(res)


if __name__ == "__main__":
    main()
