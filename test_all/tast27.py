def a(n):
    a=b=c=e=f=g=h=k=l=0
    count={str(i):0 for i in range(10)}
    for i in range(n):
        for j in str(i):
            count[j]+=1
    return count
n=int(input())
print(a(n))