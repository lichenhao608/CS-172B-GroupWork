import string
from collections import Counter


def prepross(file_name):
    '''
    This preprocesses the data with several rules:
    1) remaining the all upper cases words
    2) convert the words at the start of a sentence into lower case except one
       satisfy 1)
    3) remaining ! and ? punctuation
    '''
    file = open(file_name, 'r')
    d = Counter()
    data = []
    out = []

    for line in file:
        text, y = line.split('\t')
        y = int(y.strip())
        out.append(y)
        data.append(check(text, d))
    file.close()

    #d_set = set(d)
    '''
    for key,value in d.items():
        if key.islower() and value <= 5:
            d_set.remove(key)

    for item in data:
        for i, word in enumerate(item):
            word = item[i]
            if word not in d_set:
                item[i] = 'UNKNOWN'
    '''

    return (data, out, d)


def check(sent, word_d):
    '''
    This function works on single sentences that devide each sentence into words
    that satisfying all rules
    '''

    special = '!?'
    tor = []
    char = ''
    for c in sent:
        if c not in string.punctuation and c != ' ':
            char += c

        else:
            char = char.strip()
            if char == '':
                if c in special:
                    tor.append(c)
                    word_d.update(c)

            elif char.isdigit():
                tor.append('NUMERIC')
                word_d.update('NUMERIC')

            else:
                if char[0].isupper() and char[-1].isupper():
                    tor.append(char)
                    word_d.update(char.lower())
                else:
                    tor.append(char.lower())
                    word_d.update(char.lower())

            char = ''

    return tor


if __name__ == "__main__":
    sentence, t, vocab = prepross(
        'sentiment labelled sentences\\sentiment labelled sentences\\amazon_cells_labelled.txt')
    print(sentence)
    print(t)
