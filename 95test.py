import numpy as np

np.set_printoptions(edgeitems=3, infstr='inf',
                    linewidth=75, nanstr='nan', precision=8,
                    suppress=False, threshold=1000, formatter=None)


def f2():
    o = list(range(1, 101))
    n = np.zeros((100))
    p = np.ones((100))
    # print(o)
    # print(n)
    # print(p)
    m = 1
    l = 1
    b = 1
    # 1
    k = 1
    while k <= 100:
        # 2
        lcount = 0
        # 3
        b = b + 1
        print("k")
        print(k)
        while lcount:
            print(lcount)
            if b > 100:
                # 4
                p[b] = 1
            if p[b] == 0:
                continue
            else:
                lcount += 1
                # 5
                if not 5:
                    continue
                else:
                    n[k] = b
                    # 6
                    k += 1
    print("end")
    print(n[m])


def f3():
    n = 5
    a = np.zeros((n, n))
    k = 0
    l = 1
    while 2 * n != l:
        if l <= n:
            m = l
            i = 1
            j = l
        else:
            m = 2 * n - l
            i = l + 1 - n
            j = n
        while m > 0:
            k += 1
            if l % 2 == 0:
                a[i - 1, j - 1] = k
            else:
                a[j - 1, i - 1] = k
            i = i + 1
            j = j - 1
            m = m - 1
        l = l + 1
    print(a)


if __name__ == "__main__":
    f2()
    f3()
