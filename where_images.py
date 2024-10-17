# # for_train_val_txt.py
# # 此代码和kitti_data文件夹同目录
# import glob
# train_path = r'F:\mod\LLVIP\visible\train\images'
# test_path = r'F:\mod\LLVIP\visible\test\images'
# # video_path = 'video_data/'
# def generate_train_and_val(image_path, txt_file):
#     with open(txt_file, 'w') as tf:
#         for jpg_file in glob.glob(image_path + '*.jpg'):
#             tf.write(jpg_file + '\n')
# generate_train_and_val(train_path, 'train.txt') # 生成的train.txt文件所在路径
# generate_train_and_val(test_path, 'test.txt') # 生成的val.txt文件所在路径



# -*- coding: utf-8 -*-
# @Time :
# @Author :
# @Site :  将图片的地址写入到txt中
# @File : jpg2txt.py
# @Software: PyCharm

import os


def writejpg2txt(images_path, txt_name):
    # 打开图片列表清单txt文件
    file_name = open(txt_name, "w")
    # 将路径改为绝对路径
    images_path = os.path.abspath(images_path)
    # 查看文件夹下的图片
    images_name = os.listdir(images_path)

    count = 0
    # 遍历所有文件
    for eachname in images_name:
        # 按照需要的格式写入目标txt文件
        file_name.write(os.path.join(images_path,eachname) + '\n')
        count += 1
    print('生成txt成功！')
    print('{} 张图片地址已写入'.format(count))
    file_name.close()


if __name__ == "__main__":

    # 图片存放目录
    images_path = '/mnt/cd/ljx/m3fd/visible/train/images'
    # 生成图片txt文件命名
    txt_name = 'train.txt'
    txt_name = os.path.abspath(txt_name)
    if not os.path.exists(txt_name):
        os.system(r"touch {}".format(txt_name)) #调用系统命令行来创建文件
    #将jpg绝对地址写入到txt中
    writejpg2txt(images_path, txt_name)



# import os
#
# file_path = '/media/btbu/gt/ljx/aligned/align/visible/train/images'
# path_list = os.listdir(file_path)
#
# path_name = []
#
# for i in path_list:
#     path_name.append(i.split(".")[0])
#
# for file_name in path_name:
#     with open("train.txt", "a") as file:
#         file.write(file_name + "\n")
#         print(file_name)
#     file.close()