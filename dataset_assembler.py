import numpy as np
import os
import skimage
from PIL import Image
import random

if __name__ == '__main__':
    # Parameters
    root = "C:/Users/liuke/Mechatronics" # os.getcwd()
    train_ratio = 0.7
    valid_ratio = 0.15
    test_ratio = 0.15
    downsample_rate = 4
    output_size = 480
    ignore_factor = 0.95 # max overlap you are willing to accept
    dataset_name = "dataset3"

    # Locate all vein data directories
    print("Looking for data directories...")
    directories = os.listdir(root)
    relevant = []
    for name in directories:
        if name.endswith('_arm'):
            relevant.append(name)
    print(f"Found {relevant}")

    # Traverse the directories to merge label files and shove into dataset
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

    N = len(data_pairs)
    shuffled = random.sample(data_pairs, k = N)
    train_index = int(train_ratio * N)
    valid_index = int((train_ratio + valid_ratio) * N)

    train = shuffled[:train_index]
    valid = shuffled[train_index: valid_index]
    test = shuffled[valid_index:]

    dilation_mask = skimage.morphology.disk(5)

    dataset_path = os.path.join(root, dataset_name)
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

            # Open the image and check dimensions
            # Downprocess
            image = skimage.io.imread(old_image_path)
            small_image = image
            
            # Transpose if necessary
            if image.shape[0] > image.shape[1]:
                small_image = small_image.transpose((1, 0, 2))
                label = label.transpose()
            
            # Dilation:
            img_name = os.path.basename(pair[0])
            if img_name.find('sidney') != -1:
                label = skimage.morphology.dilation(label, footprint=dilation_mask)

            # Crop to 3968 x 1920
            # small_image = small_image[20:-20, 32:-32]
            # label = label[20:-20, 32:-32]
            # Crop to 4032 x 1920
            # small_image = small_image[20:-20, 32:-32]
            # label = label[20:-20, 32:-32]

            # Downsample
            old_size = small_image.shape
            new_size = (old_size[0] // downsample_rate, old_size[1] // downsample_rate)
            small_image = skimage.transform.resize(small_image, new_size)
            label = skimage.transform.resize(label, new_size)
            
            # Tiling
            row_progress = 0
            row_count = 0
            col_progress = 0
            col_count = 0
            max_row = new_size[0]
            max_col = new_size[1]
            
            while (row_progress < max_row):
                new_row_end = row_progress + output_size
                # If there is not enough image for another unique tile, add if overlap is not too great 4096 < 4032 + 512 * 0.9
                if max_row <= new_row_end < max_row + output_size * ignore_factor:
                    row_progress = max_row - output_size
                    new_row_end = max_row
                elif new_row_end > max_row: # too much overlap, don't make new image
                    break

                while (col_progress < max_col):
                    # Advance columns
                    new_col_end = col_progress + output_size
                    if max_col <= new_col_end < max_col + output_size * ignore_factor:
                        col_progress = max_col - output_size
                        new_col_end = max_col
                    elif new_col_end > max_col:
                        break

                    # Crop
                    crop_image = small_image[row_progress: new_row_end, 
                                             col_progress: new_col_end]
                    crop_label = label[row_progress: new_row_end, 
                                       col_progress: new_col_end]
                    
                    # Save image crop
                    img_name = os.path.basename(pair[0])[:-4] + f"_{row_count}_{col_count}" + ".jpg" 
                    new_image_path = os.path.join(sub_dir_path, img_name)
                    skimage.io.imsave(new_image_path, (crop_image * 255).astype(np.uint8))
            
                    # Save label crop
                    label_name = img_name[:-4] + "_label.npy" 
                    print(label_name, np.count_nonzero(label), crop_image.shape, crop_label.shape)
                    new_label_path = os.path.join(sub_dir_path, label_name)
                    np.save(new_label_path, crop_label)

                    col_progress += output_size
                    col_count += 1

                # Advance the rows
                row_progress += output_size
                row_count += 1

                # Reset cols
                col_progress = 0
                col_count = 0

            # skimage.io.imsave(new_image_path, (small_image * 255).astype(np.uint8))
            
            # label_name = img_name[:-4] + "_label.npy" 
            # print(label_name, np.count_nonzero(label), label.shape)
            # new_label_path = os.path.join(sub_dir_path, label_name)
            # np.save(new_label_path, label)