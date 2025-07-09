import random

print("請輸入您的戶籍地址")
adress=input()
count=['台北市', '台中市', '基隆市', '台南市', '高雄市', '新北市 (臺北縣 )', '宜蘭縣', '桃園市 ( 桃園縣 )','新竹縣', '苗栗縣', '臺中縣', '南投縣', '彰化縣', '雲林縣', '嘉義縣', '台南縣', '高雄縣', '屏東縣',
       '花蓮縣', '台東縣', '澎湖縣', '陽明山管理局	', '金門縣	', '連江縣', '嘉義市', '新竹市']
english=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N','P', 'Q', 'R', 'S','T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'I', 'O']
u=[]
number=[]
b=10
c=10
for i in count:
    b+=1
    if adress==i:
        for j in english:
            c+=1
            if c==b:
                e=j
print("請輸入您的性別")
c=input()
if c=='男':
    d=1
else:
    d=2
two=random.randint(0, 9)
three=random.randint(0,9)
four=random.randint(0, 9)
five=random.randint(0, 9)
six=random.randint(0, 9)
seven=random.randint(0, 9)
eight=random.randint(0, 9)
b=str(b)
for z in b:
    u.append(z)
english1=int(u[0])
english2=int(u[1])
nine=(english1*1+english2*9+d*8+two*7+three*6+four*5+five*4+six*3+seven*2+eight*1)%10
number.append(d)
number.append(two)
number.append(three)
number.append(four)
number.append(five)
number.append(six)
number.append(seven)
number.append(eight)
number.append(nine)
print(e, end=' ')
for f in number:
    print(f, end=' ')