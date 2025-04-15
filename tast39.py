def a(n):
    word=n.split()
    min_count=min(word, key=lambda x:len(x))
    return min_count
n=input()
print(a(n))