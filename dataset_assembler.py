import numpy as np
import os
import skimage
from PIL import Image
import random

if __name__ == '__main__':
    # Locate all vein data directories
    print("Looking for data directories...")
    directories = os.listdir()
    relevant = []
    for name in directories:
        if name.endswith('_arm'):
            relevant.append(name)
    print(f"Found {relevant}")

    # Traverse the directories to merge label files and shove into dataset
    root = os.getcwd()
    data_dict = {}
    print(f"Looking for data images with relevant labels...")
    for dir in relevant: # Go through each person's directory
        dir_path = os.path.join(root, dir)
        for filename in os.listdir(dir_path):
            if filename.endswith('.jpg'): # Found picture
                pic_path = os.path.join(dir_path, filename)
                print(f"Found picture {pic_path}...", end = '')
                # See if picture has relevant label
                index = 0
                pic_name = filename[:-4]
                label_name = os.path.join(dir_path, pic_name + "_label" + str(index) + ".npy")
                if os.path.exists(label_name):
                    print(f"found label {index}...", end = '')
                
                    label = np.load(label_name) > 0
                    print(f"{label_name} has type {label.dtype}")
                    while True: # merge any extra labels
                        index += 1
                        label_name = os.path.join(dir_path, pic_name + "_label" + str(index) + ".npy")
                        if os.path.exists(label_name):
                            print(f"found label {index}...", end = '')
                            new_label = np.load(label_name)
                            label = np.logical_or(label, new_label)
                        else:
                            break

                    # Record in dictionary
                    classes = np.zeros(label.shape)
                    classes[label] = 1.0
                    data_dict[pic_path] = classes
                    print()
                else:
                    print("No label found")

    # Shuffle and divide into train, test and valid
    images = data_dict.keys()
    labels = data_dict.values()
    data_pairs = list(zip(images, labels))

    train_ratio = 0.7
    valid_ratio = 0.15
    test_ratio = 0.15
    downsample_rate = 2
    N = len(data_pairs)
    shuffled = random.sample(data_pairs, k = N)
    train_index = int(train_ratio * N)
    valid_index = int((train_ratio + valid_ratio) * N)

    train = shuffled[:train_index]
    valid = shuffled[train_index: valid_index]
    test = shuffled[valid_index:]

    dilation_mask = skimage.morphology.disk(5)

    dataset_path = os.path.join(root, "dataset")
    if not os.path.isdir(dataset_path):
        os.mkdir(dataset_path)
    for subdir in [('train', train), ('valid', valid), ('test', test)]:
        sub_dir_path = os.path.join(dataset_path, subdir[0])
        if not os.path.isdir(sub_dir_path):
            os.mkdir(sub_dir_path)

        for pair in subdir[1]:
            # image path
            old_image_path = pair[0]
            label = pair[1]
            img_name = os.path.basename(pair[0])
            new_image_path = os.path.join(sub_dir_path, img_name)

            # Open the image and check dimensions
            # Downprocess
            image = skimage.io.imread(old_image_path)
            small_image = image
            
            # Transpose if necessary
            if image.shape[0] > image.shape[1]:
                small_image = small_image.transpose((1, 0, 2))
                label = label.transpose()
            
            # Dilation:
            if img_name.find('sidney') != -1:
                label = skimage.morphology.dilation(label, footprint=dilation_mask)

            # Crop to 1920
            small_image = small_image[20:-20, 32:-32]
            label = label[20:-20, 32:-32]

            # Downsample by 2
            old_size = small_image.shape
            new_size = (old_size[0] // downsample_rate, old_size[1] // downsample_rate)
            small_image = skimage.transform.resize(small_image, new_size)

            label = skimage.transform.resize(label, new_size)

            skimage.io.imsave(new_image_path, (small_image * 255).astype(np.uint8))
            
            label_name = img_name[:-4] + "_label.npy" 
            print(label_name, np.count_nonzero(label), label.shape)
            new_label_path = os.path.join(sub_dir_path, label_name)
            np.save(new_label_path, label)