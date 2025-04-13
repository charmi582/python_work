def a(n):
    count={}
    for i in n:
        if i in count:
            count[i]+=1
        else:
            count[i]=1
    result=[k for k, v in count.items() if v%2==1 ]
    return result
n=input()
print(a(n))
