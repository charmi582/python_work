def a(n):
    word=n.split()
    count={}
    for i in word:
        l=len(i)
        if l not in count:
            count[l]={}
            count[l][i]=count[l].get(i, 0)+1
    return count
n=input()
print(a(n))