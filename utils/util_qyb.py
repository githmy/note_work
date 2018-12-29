# -*- coding: utf-8 -*-
import json, os
from django.shortcuts import HttpResponse


def dic_name(instr):
    file_posi = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/static/json/dic_name.json"
    f = file(file_posi)
    lineline = f.read()
    f.close()
    setting = json.loads(lineline)
    if instr in setting:
        return setting[instr]
    else:
        return instr


def dic_online(instr):
    file_posi = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/static/json/dic_online.json"
    f = file(file_posi)
    lineline = f.read()
    f.close()
    setting = json.loads(lineline)
    if instr in setting:
        return setting[instr]
    else:
        return instr
