# coding:utf-8
import numpy as np
import codecs


class Fitfun(object):
    def __init__(self):
        self.sum_mse = 0.0

    def write_result(self):
        "write result"
        with codecs.open("result.txt", 'w', 'utf-8') as f:
            f.write(str(self.sum_mse))

    def readfile2line(self, filename, xl, xr, yl, yh):
        "read file"
        text = codecs.open(filename, 'r', 'utf-8').read()
        xydata = [i1.strip().split() for i1 in text.strip().split("\n")]
        xydata = [[float(i1[0]), float(i1[1])] for i1 in xydata]
        xydata = [i1 for i1 in xydata if i1[0] >= xl and i1[0] <= xr and i1[1] >= yl and i1[1] <= yh]
        xydata = np.array(list(zip(*xydata)))
        # print(xydata)
        # print(xydata[0, 1:])
        # print(xydata[0, 0:-1])
        tvalue = sum(xydata[0, 1:] - xydata[0, 0:-1])
        if tvalue < 0:
            xydata[0] = np.flipud(xydata[0])
            xydata[1] = np.flipud(xydata[1])
        return xydata

    def _interpolate_error(self, xydata1, xydata2):
        y_list = self._interpolate_value(xydata1, xydata2)
        mse = 0.0
        for i1, i2 in zip(y_list, xydata2[1, :]):
            mse += pow((i1 - i2), 2)
        return mse

    def _interpolate_value(self, xydata1, xydata2):
        len1 = len(xydata1[0])
        y_interpolate = []
        for i2 in range(len(xydata2[0])):
            l_index = 0
            r_index = 0
            if xydata2[0, i2] < xydata1[0, 0]:
                pass
            elif xydata2[0, i2] > xydata1[0, len1 - 1]:
                l_index = len1 - 1
                r_index = len1 - 1
            else:
                for i1 in range(len1 - 1):
                    if xydata2[0, i2] >= xydata1[0, i1] and xydata2[0, i2] < xydata1[0, i1 + 1]:
                        l_index = i1
                        r_index = i1 + 1
            bflin = xydata1[1, l_index] + (xydata2[0, i2] - xydata1[0, l_index]) * (
                xydata1[1, r_index] - xydata1[1, l_index]) / (xydata1[0, r_index] - xydata1[0, l_index])
            y_interpolate.append(bflin)
        return y_interpolate

    def fit2lines(self, injson):
        """
        we suppose file1 as experment data
        """
        mse = 0.0
        for i1 in injson:
            xl = i1["xrange1"]
            xr = i1["xrange2"]
            yl = i1["yrange1"]
            yh = i1["yrange2"]
            logstr = i1["xylog"]
            xydata1 = self.readfile2line(i1["file1"], xl, xr, yl, yh)
            xydata2 = self.readfile2line(i1["file2"], xl, xr, yl, yh)
            if logstr == "xlog":
                xydata1[0] = np.log10(xydata1[0])
                xydata2[0] = np.log10(xydata2[0])
            elif logstr == "ylog":
                xydata1[1] = np.log10(xydata1[1])
                xydata2[1] = np.log10(xydata2[1])
            elif logstr == "xylog":
                xydata1[0] = np.log10(xydata1[0])
                xydata2[0] = np.log10(xydata2[0])
                xydata1[1] = np.log10(xydata1[1])
                xydata2[1] = np.log10(xydata2[1])
            elif logstr == "linear":
                pass
            else:
                pass
            # print("interpolate_error", self._interpolate_error(xydata1, xydata2))
            mse += i1["weight"] * self._interpolate_error(xydata1, xydata2)
        return mse

    def line_lastvalue(self, injson):
        mse = 0.0
        for i1 in injson:
            xl = i1["xrange1"]
            xr = i1["xrange2"]
            yl = i1["yrange1"]
            yh = i1["yrange2"]
            logstr = i1["xylog"]
            xydata1 = self.readfile2line(i1["file1"], xl, xr, yl, yh)
            xydata2 = self.readfile2line(i1["file2"], xl, xr, yl, yh)
            if logstr == "xlog":
                xydata1[0] = np.log10(xydata1[0])
                xydata2[0] = np.log10(xydata2[0])
            elif logstr == "ylog":
                xydata1[1] = np.log10(xydata1[1])
                xydata2[1] = np.log10(xydata2[1])
            elif logstr == "xylog":
                xydata1[0] = np.log10(xydata1[0])
                xydata2[0] = np.log10(xydata2[0])
                xydata1[1] = np.log10(xydata1[1])
                xydata2[1] = np.log10(xydata2[1])
            elif logstr == "linear":
                pass
            else:
                pass
            usedata1, usedata2 = [xydata1[0], xydata2[0]] if i1["value_at"] == "x" else [xydata1[1], xydata2[1]]
            tmse = 0.0
            if i1["value_type"] == "last":
                tmse = pow((usedata1[-1] - usedata2[-1]), 2)
            elif i1["value_type"] == "first":
                tmse = pow((usedata1[0] - usedata2[0]), 2)
            else:
                pass
            # print("last2: ", usedata1[-1], usedata2[-1])
            mse += i1["weight"] * tmse
        return mse

    def get_vth_error(self, injson):
        mse = 0.0
        for i1 in injson:
            xydata = self.readfile2line(i1["file2"], -1e49, 1e49, -1e49, 1e49)
            txydata = (xydata[1, 1:] - xydata[1, 0:-1]) / (xydata[0, 1:] - xydata[0, 0:-1])
            index1 = np.argmax(np.abs(txydata[0:-1]))
            vth = xydata[0, index1] - xydata[1, index1] / txydata[index1]
            # print("vth", vth)
            tmse = pow((i1["value1"] - vth), 2)
            mse += i1["weight"] * tmse
        return mse


def main():
    # 1. initial
    ff = Fitfun()
    # 2. define your error
    # self_define(ff)  # move here for debug
    try:
        self_define(ff)  # move here for record
    except Exception as e:
        ff.sum_mse += 1e99
    # 3. last write ff.sum_mse to defined file
    ff.write_result()


def self_define(cls):
    """ core function"""
    ff = cls
    # 1. example: fitting curves. such as leakage trend.
    # xylog：xlog,ylog,xylog,linear
    fit2lines_json = [
        {"file1": "meas_Vbg_5.txt", "file2": "vbg_5.txt", "xylog": "ylog",
         "weight": 1, "xrange1": -1e19, "xrange2": 0, "yrange1": -1e19, "yrange2": 1e19},
        {"file1": "meas.txt", "file2": "vbg0.txt", "xylog": "ylog",
         "weight": 1, "xrange1": -1e19, "xrange2": 0, "yrange1": -1e19, "yrange2": 1e19},
        {"file1": "meas_Vbg5.txt", "file2": "vbg5.txt", "xylog": "ylog",
         "weight": 1, "xrange1": -1e19, "xrange2": 0, "yrange1": -1e19, "yrange2": 1e19},
    ]
    mse1 = ff.fit2lines(fit2lines_json)
    # print("mse1:", mse1)
    ff.sum_mse += mse1
    # 2. example: fitting some value in curves. such as bv
    fit2lines_json = [
        {"file1": "meas_Vbg_5.txt", "file2": "vbg_5.txt", "xylog": "linear", "value_at": "x", "value_type": "last",
         "weight": 1, "xrange1": -1e19, "xrange2": 1e19, "yrange1": -1e19, "yrange2": 1e19},
    ]
    mse1 = ff.line_lastvalue(fit2lines_json)
    # print("mse1:", mse1)
    ff.sum_mse += mse1
    # 3. example: fitting some value in curves with target value. such as vth.
    fit2lines_json = [
        {"value1": 0.7, "file2": "vt.dat", "weight": 1},
    ]
    mse1 = ff.get_vth_error(fit2lines_json)
    # print("mse1:", mse1)
    ff.sum_mse += mse1
    print("sum_mse:", ff.sum_mse)
    # 4. you can define more target error to minimize.


if __name__ == '__main__':
    main()
