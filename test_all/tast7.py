a='一'
b='二'
c='三'
d='四'
e='五'
f='六'
g='七'
h='八'
j='九'
def convert_to_chinese(num: str):
    k=[]

    for i in num:
        if i=='1':
            k.append(a)
        elif i=='2':
            k.append(b)
        elif i=='3':
            k.append(c)
        elif i=='4':
            k.append(d)
        elif i=='5':
            k.append(e)
        elif i=='6':
            k.append(f)
        elif i=='7':
            k.append(g)
        elif i=='8':
            k.append(h)
        elif i=='9':
            k.append(j)
    return(''.join(k))
num=input()
print(convert_to_chinese(num))
    
    
