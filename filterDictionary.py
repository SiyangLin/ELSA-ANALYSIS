# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 16:55:11 2019

@author: 39695
"""

import re

def getVariableWithNotations(path):
    myPattern = re.compile(r'Variable = }{\\f2\\fs20\\cf4 .+?	}{\\b\\cf1 Variable label = }{\\cf4 .+?\\par')
    result = None
    with open(path, 'r', encoding='utf-8') as f:
        text = f.readlines()
        result = myPattern.findall(text[0])
    result = [s.replace(' }{\\f2\\fs20\\cf4 ','').replace('}{\\cf4 ','') \
    .replace('\\par','').replace('}{\\b\\cf1','').replace('Variable label =',';') \
    .replace('Variable =','') for s in result]
    variableDict = {s.split(';')[0].rstrip():s.split(';')[1].lstrip() for s in result}
    return variableDict

def saveSelectedAttributesWithNotation(dic, df, path):
    popList = [k for k in dic if k not in df.columns]
    for k in popList:
        dic.pop(k)
    
    with open(path, 'w', encoding='utf-8') as f:
        for k,v in dic.items():
            f.write(k)
            f.write(':')
            f.write(v)
            f.write('\n')

if __name__ == '__main__':
    d = getVariableWithNotations('../wave_5_elsa_data_v4_ukda_data_dictionary.rtf')