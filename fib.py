a = [1, 2]
while a[-1] < 100000000:
    a.append(a[-1] + a[-2])

N = int(input())
for case in range(N):
    num = int(input())
    print(f"{num} = ", end='')

    found = 0 
    
    for f in a[::-1]:
        if num >= f:
            num -= f
            found = 1  
            print(1, end='')
        elif found:    
            print(0, end='')

    print(" (fib)")