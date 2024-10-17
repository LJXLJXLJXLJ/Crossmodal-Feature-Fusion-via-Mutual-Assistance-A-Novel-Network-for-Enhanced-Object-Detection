import pickle

# 读取.pkl文件
with open('/media/btbu/gt/ljx/aligned/align/infrared/data.pkl', 'rb') as f:
    data = pickle.load(f)
with open('/media/btbu/gt/ljx/aligned/align/visible/data.pkl', 'rb') as g:
    data2 = pickle.load(g)
# 使用读取的数据
lst=[]
lst2=[]
lst3=[]
lst4=[]
for i in data:
    print(i)
    lst.append(i)
for j in data2:
    print(i)
    lst2.append(j)
for i in lst:
    lst3.append(i[-22:-17])
for i in lst2:
    lst4.append(i[-13:-8])
diff_elements = list(set(lst3) - set(lst4)) + list(set(lst4) - set(lst3))#     b.write(data.__str__())
print(diff_elements)