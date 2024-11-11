import os
import shutil

source_path = '/home/ai/pythonproject/catvsdog'
train_path = os.path.join(source_path, 'data','train')
test_path = os.path.join(source_path, 'data','test')


os.makedirs(os.path.join(train_path,'cat'),exist_ok=True)
os.makedirs(os.path.join(train_path,'dog'),exist_ok=True)
os.makedirs(os.path.join(test_path,'cat'),exist_ok=True)
os.makedirs(os.path.join(test_path,'dog'),exist_ok=True)

split_ratio = 0.8

for category in ["cat","dog"]:
    categer_path = os.path.join(source_path,'animals',category)
    images = os.listdir(categer_path)

    split_idx = int(len(images)*split_ratio)

    train_images = images[:split_idx]
    test_images = images[split_idx:]

    for image in train_images:
        shutil.copyfile(os.path.join(categer_path,image),os.path.join(train_path,category,image))

    for image in test_images:
        shutil.copyfile(os.path.join(categer_path,image),os.path.join(test_path,category,image))


