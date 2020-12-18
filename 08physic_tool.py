import scipy.constants as sc


def main():
    print("电荷数：{}".format(sc.elementary_charge))
    print("普朗克常数：{}".format(sc.h))
    print("光速：{}".format(sc.c))
    lambda1 = 1.55e-6
    v = sc.c / lambda1
    ev_band = sc.h * v / sc.elementary_charge
    print("ev_band {} at lambda = {}".format(ev_band, lambda1))


if __name__ == '__main__':
    main()
