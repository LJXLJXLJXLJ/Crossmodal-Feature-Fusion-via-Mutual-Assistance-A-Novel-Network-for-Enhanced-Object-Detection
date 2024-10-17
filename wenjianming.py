import os

folder_path = r"/media/btbu/gt/ljx/aligned/align/visible/val/labels"
file_list = os.listdir(folder_path)

for file_name in file_list:
    # 旧文件路径
    old_path = os.path.join(folder_path, file_name)
    # 新文件名
    new_file_name = file_name[0:11]+"RGB"+'.txt'
    # 新文件路径
    new_path = os.path.join(folder_path, new_file_name)
    # 修改文件名
    os.rename(old_path, new_path)
    print(new_file_name)
