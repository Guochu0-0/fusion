# coding:utf-8
import os

'''
    为数据集生成对应的txt文件
'''

train_txt_path = os.path.join("..", "..", "Data", "train.txt")
train_ir_dir = os.path.join("..", "..", "Data", "Train_ir")
train_vi_dir = os.path.join("..", "..", "Data", "Train_vi")

test_txt_path = os.path.join("..", "..", "Data", "test.txt")
test_ir_dir = os.path.join("..", "..", "Data", "Test_ir")
test_vi_dir = os.path.join("..", "..", "Data", "Test_vi")


def gen_txt(txt_path, ir_img_dir, vi_img_dir):
    f = open(txt_path, 'w')

    ir_img_list = os.listdir(ir_img_dir)
    vi_img_list = os.listdir(vi_img_dir)
    for i in range(len(ir_img_list)):
        ir_img_path = os.path.join(ir_img_dir, ir_img_list[i])
        vi_img_path = os.path.join(vi_img_dir, vi_img_list[i])
        line = vi_img_path + ' ' + ir_img_path + '\n'
        f.write(line)

    f.close()


if __name__ == '__main__':
    gen_txt(train_txt_path, train_ir_dir, train_vi_dir)
    gen_txt(test_txt_path, test_ir_dir, test_vi_dir)
