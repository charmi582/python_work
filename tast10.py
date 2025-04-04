def a(f: str):
    b=''
    c='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i in f:

            if i in c:
                b+=i
                break
    return b

f=input()
print(a(f))