print("請輸入您的戶籍地址")
adress=input()
count=['台北市', '台中市', '基隆市', '台南市', '高雄市', '新北市 (臺北縣 )', '宜蘭縣', '桃園市 ( 桃園縣 )','新竹縣', '苗栗縣', '臺中縣', '南投縣', '彰化縣', '雲林縣', '嘉義縣', '台南縣', '高雄縣', '屏東縣',
       '花蓮縣', '台東縣', '澎湖縣', '陽明山管理局	', '金門縣	', '連江縣', '嘉義市', '新竹市']
english=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N','P', 'Q', 'R', 'S','T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'I', 'O']
u=[]
number=[]
b=10
c=10
m=[]
for i in count:
    b+=1
    if adress==i:
        for j in english:
            c+=1
            if c==b:
                e=j
print("請輸入您身分證號碼開頭")
o=input()
for z in range(2):
    if o!=e:
        print("身分證號碼不正確")
        break


for z in range(9):
    print(f"請輸入您第{z}個身分證數字")
    b=eval(input())
    u.append(b)
b=str(b)
for v in b:
    m.append(v)
first=int(m[0])
two=int(m[1])
result=(first*1+two*9+u[1]*8+u[2]*7+u[3]*6+u[4]*5+u[6]*4+u[7]*3+u[8]*2+u[9]*1)%10
if result==u[10]:
    print("正確身分證號碼")
else:
    print("不正確")
