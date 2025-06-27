def a(n):
    word=n.split()
    count={}
    for i in word:
        l=len(i)
        if l not in count:
            count[l]=[]
        if i not in count[l]:
            count[l].append(i)
    return count
n=input()
print(a(n))


