import pandas

df = pandas.read_excel("ONET Excel Files 29.3/Skills.xlsx")
ls = []
for i in list(df["Element Name"]):
    if i not in ls:
        ls.append(i)

print(len(ls))
print(ls)


# 879 Jobs
# 35 Skills