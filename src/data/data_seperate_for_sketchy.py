import os


sketchy_root = 'C:/Users/xiangjun/datafolder/sketch_image/rendered_256x256/256x256\photo'
used_dir = 'tx_000100000000'

with open('zeroshot_classes_sketchy.txt') as f:
    test_classes = f.read().splitlines()
    total_classes = os.listdir(os.path.join(sketchy_root,used_dir))
    for test_class in test_classes:
        total_classes.remove(test_class)
total_classes = [line + '\n' for line in total_classes]
with open('sketchy_for_train.txt','a') as ff:
    ff.writelines(total_classes)