def a(f: str):
    b=''
    c='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i in f:
        for j in c:
            if i==c:
                b+=i
                break
    return b

f=input()
print(a(f))