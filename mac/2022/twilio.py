def imageval(url, msize):
    if msize == 'none':
        maxsize = 1000000
    maxsize = 0
    unit = msize[-2:].lower()
    size = int(msize[:-2])
    if unit == 'kb':
        maxsize = int(size) * 1000
    if unit == 'mb':
        maxsize = int(size) * 1000000
    if unit == 'gb':
        maxsize = int(size) * 1000000000
    for i in url:
        if int(i[1]) > maxsize:
            i[1] = 'False'
        else:
            i[1] = 'True'
    return url

