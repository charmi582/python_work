def a(n):
    b=[]
    for y in range(n):
        for i in str(n):
            for j in i:
                if int(j)%2==1:
                    b.append(y)
    return b
n=int(input())
print(a(n))