import os
import re
import sys
import time
import psutil
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_curve(x, ys, titles):
    # titles = ["资金池增量", "双头补金额增量", "借贷增量", "工资增量", "创始人收益增量"]
    # ys = [fs_ins.y_capital_change, fs_ins.y_subsidy_change, fs_ins.y_borrow_change, fs_ins.y_salary_change,
    #       fs_ins.y_creater_change]
    # plot_curve(fs_ins.x_label, ys, titles)
    yins = [np.array(y) for y in ys]
    xin = np.arange(0, len(ys[0]))
    nums = len(ys)
    colors = ["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#ff00ff", "#00ffff", "#000000"] * (nums // 7 + 1)
    # 长 宽 背景颜色
    plt.figure(figsize=(12, 6), facecolor='w')
    # plt.figure(facecolor='w')
    # print(xin, yins[0],colors[0],titles[0])
    for n in range(nums):
        plt.plot(xin, yins[n], color=colors[n], linestyle='-', linewidth=1.2, marker="", markersize=7,
                 markerfacecolor='b', markeredgecolor='g', label=titles[n])
        plt.legend(loc='upper right', frameon=False)
    # plt.plot(xin, yin, color='r', linestyle='-', linewidth=1.2, marker="*", markersize=7, markerfacecolor='b',
    #          markeredgecolor='g')
    plt.xlabel("x", verticalalignment="top")
    plt.ylabel("y", rotation=0, horizontalalignment="right")
    # xticks = ["今天", "周五", "周六", "周日", "周一"]
    # show_inte = 30
    show_inte = 7
    s_xin = [i1 for i1 in xin if i1 % show_inte == 0]
    s_x = [i1 for id1, i1 in enumerate(x) if id1 % show_inte == 0]
    plt.xticks(s_xin, s_x, rotation=90, fontsize=10)
    # plt.xticks(xin, x, rotation=90, fontsize=5)
    # yticks = np.arange(0, 500, 10)
    # plt.yticks(yticks)
    # plt.title(title)
    # plt.grid(b=True)
    interval = 3  # polling seconds
    plt.pause(interval)
    plt.close()


def main(processname):
    # 1. 启动指定进程
    pids = psutil.pids()
    pidlist = []
    for pid in pids:
        p = psutil.Process(pid)
        # print('pid-%s,pname-%s' % (pid, p.name()))
        if re.match(f"^{processname}", p.name()):
            print('find pid: %s, pname: %s' % (pid, p.name()))
            pidlist.append(pid)
            # cmd = 'taskkill /F /IM dllhost.exe'
            # os.system(cmd)
    # 2. get process 信息
    if len(pidlist) > 1:
        print(f"存在多个{processname}进程!")
        exit()
    elif len(pidlist) == 0:
        print(f"没有找到{processname}进程!")
        exit()
    pid = pidlist[0]
    p = psutil.Process(pid)
    # monitor process and write data to file
    interval = 3  # polling seconds
    titles = ["cpu", "mem"]
    x, ys = [], [[], []]
    savefile = os.path.join("..", "process_monitor_" + p.name() + '_' + str(pid) + ".csv")
    with open(savefile, "a+") as f:
        f.write("time,cpu%,mem%\n")  # titles
        try:
            while True:
                current_time = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
                cpu_percent = p.cpu_percent()
                mem_percent = p.memory_percent()
                # print(cpu_percent ,mem_percent)
                x.append(current_time)
                ys[0].append(float(cpu_percent))
                ys[1].append(float(mem_percent))
                line = current_time + ',' + str(cpu_percent) + ',' + str(mem_percent)
                f.write(line + "\n")
                f.flush()
                time.sleep(interval)
                # 3. 绘图
                # plot_curve(x, ys, titles)
        except Exception as e:
            pass
    titles = ["mem"]
    plot_curve(x, [ys[1]], titles)
    df = pd.read_csv(savefile)
    df.set_index("time", drop=True, inplace=True)
    df["mem%"].plot()
    plt.show()


if __name__ == '__main__':
    # savefile = os.path.join("..", "process_monitor_apsys.exe_69004.csv")
    # df = pd.read_csv(savefile)
    # print(df)
    # df.set_index("time", drop=True, inplace=True)
    # df["mem%"].plot()
    # plt.show()
    # exit()
    # processname = "SimuApsys"
    processname = "apsys"
    main(processname)
