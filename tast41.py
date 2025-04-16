def a(n):
    count={}
    word=n.split()
    for i in word:
        if i in count:
            count[i]+=1
        else:
            count[i]=1
    result=[k for k, v in count.items() if v==1]
    return result
n=input()
print(a(n))