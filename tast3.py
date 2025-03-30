lines = [] 
while True:
    line = input()
    if line == "":
        break
    lines.append(line)
max_len = max(len(line) for line in lines)
lines = [line.ljust(max_len) for line in lines]
linepro = []
for col in range(max_len):
    new_line = ""
    for row in range(len(lines)):
        new_line += lines[row][col]
    linepro.append(new_line)
for line in linepro:
    print(line)
