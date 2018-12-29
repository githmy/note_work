# -*- coding: utf-8 -*-

import math

# log归一化
def log_normalize(a):
    s=0
    for x in a:
        s+=x
    if s==0:
        print "Error,, from log_ normalize"
        return
    s= math.log(s)
    for i in range(len (a)):
        if a[i]==0:
            a[i]= infinite
        else:
            a[i]= math.log(a[i])-s





