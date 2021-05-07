#!/usr/bin/python

import os,re,csv
import sys
import numpy as np
from tqdm import tqdm
"""
产生sample.csv文件，供训练使用。
"""


def list_pictures(directory):
    imgTpe = 'JPEG'
    imgTpes = ['.jpeg','.jpg','.png', '.bmp']
    # img_list = [os.path.join(root, f) for root, _, files in os.walk(directory) for f in files if re.match(r'([\w]+\.(?:' + imgTpe + '))', f)]

    img_list = [os.path.join(root, f) for root, _, files in os.walk(directory) for f in files if
                os.path.splitext(os.path.basename(f))[-1].lower() in imgTpes]

    img_list = [str(ele).replace('\\', '/')  for ele in img_list]

    print("img_list: ", len(img_list))

    return img_list


def get_negative_images(all_images, image_names):
    num_neg_images = 1
    random_numbers = np.arange(len(all_images))
    np.random.shuffle(random_numbers)
    if int(num_neg_images) > (len(all_images) - 1):
        num_neg_images = len(all_images) - 1

    neg_count = 0
    negative_images = []

    for random_number in list(random_numbers):
        if all_images[random_number] not in image_names:
            negative_images.append(all_images[random_number])
            neg_count += 1
            if neg_count > (int(num_neg_images) - 1):
                break

    return negative_images


def get_positive_images(image_name, image_names):
    num_pos_images = 1
    random_numbers = np.arange(len(image_names))
    np.random.shuffle(random_numbers)
    if int(num_pos_images) > (len(image_names) - 1):
        num_pos_images = len(image_names) - 1

    pos_count = 0
    positive_images = []

    for random_number in list(random_numbers):
        if image_names[random_number] != image_name:
            positive_images.append(image_names[random_number])
            pos_count += 1
            if int(pos_count) > (int(num_pos_images) - 1):
                break

    return positive_images


def triplet_sampler(directory_path, saple_csvPath):

    classes = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
    all_images = []
    for class_ in classes:
        all_images += (list_pictures(os.path.join(directory_path, class_)))

    i = 1
    triplets = []
    for class_ in tqdm(classes):
        image_names = list_pictures(os.path.join(directory_path, class_))
        for image_name in image_names:
            # image_names_set = set(image_names)
            query_image = image_name
            positive_images = get_positive_images(image_name, image_names)
            for positive_image in positive_images:
                negative_images = get_negative_images(all_images, set(image_names))
                for negative_image in negative_images:

                    triplets.append([class_, query_image, positive_image, negative_image])


        if i % 10 == 0:
            print("Now processing {}th class".format(i))
        i += 1

    print("==> Sampling Done ... Now Writing ...")

    f = open(saple_csvPath,'w', newline='')
    f_csv = csv.writer(f)
    header = ['classid', 'class', 'query', 'inclass', 'outclass']
    f_csv.writerow(header)

    for ele in triplets:
        class_name = ele[0]
        class_id = str(classes.index(class_name))
        new_ele = [class_id] + [str(x).replace(directory_path, '') for  x in ele]
        f_csv.writerow(new_ele)



def test():
    """Triplet Sampling."""

    directory_path = "../data/tiny-imagenet-200/train/"
    output_directory = './triplets_new.csv'
    # triplet_sampler(directory_path, output_directory)

    import pandas as pd

    df = pd.read_csv(output_directory)
    print(df)



if __name__ == '__main__':
    # directory_path = "../data/tiny-imagenet-200/train/"
    # output_csvPath = './triplets_new.csv'
    # directory_path = sys.argv[1]
    # output_csvPath = sys.argv[2]
    # triplet_sampler(directory_path, output_csvPath)

    test()
