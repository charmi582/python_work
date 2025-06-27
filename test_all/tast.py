a=input()
n=0
for i in a:
    if i=='"':
        n+=1
        if n%2==1:
            print("``", end=" ")
        else :
            print('"', end=" ")
    else:
        print(i, end=" ")