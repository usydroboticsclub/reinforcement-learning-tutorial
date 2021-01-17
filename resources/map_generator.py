def unikeyToMaze(unikey):
    ls = []

    for i in range(25):
        ls.append(0)


    key = unikey
    letters = key[0:4]
    digits = key[4:9]

    alp = list('abcdefghijklmnopqrstuvwxyz')

    i = 0
    for letter in letters:
        if letter == 'z':
            pos = 0
        else:
            pos = alp.index(letter)
        if int(digits[i]) == 0:
            ls[pos] += 10
        else:
            ls[pos] += int(digits[i])
        i += 1

    map = []

    i = 0
    row = []
    while i <= 24:
        row.append(ls[i])
        if (i+1) % 5 == 0:
            map.append(row)
            row = []
        i += 1
    
    return (map, 2, 2, 5)
