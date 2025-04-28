def a(n):
    word=n.split()
    count={}
    for i in word:
        long=len(i)
        if long not in count:
            count[long]={}
        count[long][i]=count[long].get(i, 0)+1
    return count
n=input()
print(a(n))