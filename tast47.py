def a(n):
    word=n.split()
    count={}
    for i in word:
        count[i]=count.get(i, 0)+1
    result=sorted(count.items(), key=lambda x:[x], reverse=True)
    return result
n=input()
print(a(n))