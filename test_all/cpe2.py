n=input()
a=0
c=0
e=0
b='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
m='0123456789'
k='!@#$%^&*()+'
for i in n:
    for j in b:
        if i==j:
            a+=1
    for z in m:
        if i==z:
            c+=1
    for y in k:
        if i==y:
            e+=1
print(f"英文字母:{a}")
print(f"數字:{c}")
print(f"其他符號:{e}")
    