import numpy as np
import string
from collections import defaultdict

def prepross(file_name):
    file = open(file_name,'r')
    d = defaultdict(int)
    data = []
    out = []

    for line in file:
        text, y = line.split('\t')
        y = int(y.strip())
        out.append(y)
        data.append(check(text,d))
    file.close()

    d_set = set(d)
    for key,value in d.items():
        if key.islower() and value <= 5:
            d_set.remove(key)
    
    for item in data:
        for i in range(len(item)):
            word = item[i]
            if word not in d_set:
                item[i] = 'UNKNOWN'

    return (data,out)

def check(sent,word_d):
    special = '!?'
    tor = []
    char = ''
    for c in sent:
        if c not in string.punctuation:
            char += c
        else:
            char = char.strip()

            if char.isdigit():
                tor.append('NUMERIC')
                word_d['NUMERIC'] += 1
            else:
                if char[0].isupper() and char[-1].isupper():
                    tor.append(char)
                    word_d[char] += 1
                else:
                    tor.append(char.lower())
                    word_d[char.lower()] += 1

            if c in special:
                tor.append(c)
                word_d[c] += 1
                
            char = ''
    return tor
