def a(n):
    count={str(i):0 for i in range(10)}
    for i in str(n):
        count[i]+=1
    return count
n=int(input())
print(a(n))
