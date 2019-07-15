# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os, sys, getopt
from utils.parse_args import parseArgs


def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print('test.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    print('输入的文件为：', inputfile)
    print('输出的文件为：', outputfile)


def namespace_way():
    from argparse import Namespace
    args = Namespace(
        seed=1234,
        dropout_p=0.1,
    )
    print(args.seed, args.dropout_p)


if __name__ == "__main__":
    main(sys.argv[1:])

    # 2. 特殊格式操作
    # args = parseArgs(sys.argv)
    # del sys.argv[0]
    # os.system("python3 " + args.func + ".py " + cmdstr)

    os.system("mv ./test ./rename_test")
