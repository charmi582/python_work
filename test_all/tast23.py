a=list(map(int, input().split(" ")))
a.sort()
print(a)
if a[0]+a[1] <=a[2]:
    print("No")
elif a[0]*a[0] + a[1]*a[1]<a[2]*a[2]:
    print("Obtuse triangle")
elif a[0]*a[0] + a[1]*a[1]==a[2]*a[2]:
    print("Right triangle")
elif a[0]*a[0] + a[1]*a[1]>a[2]*a[2]:
    print("Acute triangle")