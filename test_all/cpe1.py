n=int(input())
c=0
b=0
for i in str(n):
    l=int(i)
    if l%2==0:
        b+=1
    else:
        c+=1
print(f"奇數有:{c}個")
print(f"偶數有:{b}個")