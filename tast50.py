def a(n):
    word = n.split()
    count = {}
    for i in word:
        length = len(i)
        count[length] = count.get(length, 0) + 1

    max_len = max(count, key=lambda x: count[x])

    result = []
    for w in word:
        if len(w) == max_len and w not in result:
            result.append(w)

    return result

n = input()
print(a(n))
