a=eval(input())
for i in range(2,a-1):
    
    if a%i==0:
        print(f"{a} 不為質數")
        break
else:
        print(f"{a} 是質數")
