def is_prime(n):
    for i in range(2, n-1):
        if n%i==0:
            return(f"{n} 不是質數")
            break
    else:
        return(f"{n} 視質數")
n=eval(input())
print(is_prime(n))
z=[]
for i in range(2, 101):
        for j in range(2, i):
            if i%j==0  :
                break
        else:
            z.append(i)
 
print(z)