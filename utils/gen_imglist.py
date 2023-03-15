import os
import argparse


def generate_imglist(data_path):
    """Generate imagelist for dataset.
    The dataset format is:
        - data_path
            - train
                - class0
                    - *.png / *.jpg
                - class1
                    ...
                ...
            - val (optional)
                ...
            - test (optional)
                ...
    The imglist format is:
        relative_path label (for instance: train/bird/0.jpg 0, where the label will be saved as index of classid)
        ...
    """
    split_name = os.listdir(data_path)
    if len(split_name) == 0:
        print(f"{data_path} is empty.")
        return
    for split in split_name:
        if '.' in split:
            continue
        split_path = os.path.join(data_path, split)
        class_names = os.listdir(split_path)
        img_paths = []
        labels = []
        for class_name in class_names:
            img_names = os.listdir(os.path.join(data_path, split, class_name))
            for name in img_names:
                img_path = os.path.join(split, class_name, name)
                img_paths.append(img_path)
                labels.append(str(class_names.index(class_name)))
        fw = open(os.path.join(data_path, split + '_label.txt'), 'w')
        for i in range(len(img_paths)):
            fw.write(img_paths[i] + ' ' + labels[i] + '\n')
        fw.close()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate image lists')
    parser.add_argument('data_path', type=str, default='Root path of dataset')
    args = parser.parse_args()
    generate_imglist(args.data_path)