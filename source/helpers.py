import random
import albumentations as A
import cv2
import numpy as np
import os
import shutil
from matplotlib import pyplot as plt 
from ultralytics import YOLO

category_id_to_name = { 0: "logo"}


# to load images ########################################################
def img_loader(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    ratio = width/height
    return image, height, width, ratio

# to lead yolo annotations ########################################################
def load_yolo_annotations(annotation_file):
    yolo_anno = []
    category_ids = []
    with open(annotation_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if(line[1] == " "):
                yolo_anno.append(list(map(float, line.strip().split()[1:])))
                category_ids.append(int(line[0]))
            else:
                yolo_anno.append(list(map(float, line.strip().split()[1:])))
                category_ids.append(int(line[0:2]))
    return yolo_anno, category_ids

# visualize images with annotations on them ########################################################
def visualize_bbox(img, bbox, class_name, color=(255, 0, 0), thickness=2):
    """Visualizes a single bounding box on the image"""
    
    height, width, _ = img.shape
    
    x_cen_norm, y_cen_norm, w_norm, h_norm = bbox[0], bbox[1], bbox[2], bbox[3]
    x_cen, y_cen, w, h = int(x_cen_norm*width), int(y_cen_norm*height), int(w_norm*width), int(h_norm*height)

    x_min, x_max, y_min, y_max = x_cen-w//2, x_cen+w//2, y_cen-h//2, y_cen+h//2
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=(255, 255, 255), 
        lineType=cv2.LINE_AA,
    )
    return img

def visualize(image, annotations, category_ids, category_id_to_name):
    img = image.copy()
    for idx, bbox in enumerate(annotations):
        class_name = category_id_to_name[category_ids[idx]]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)

# define augmenation pipeline ########################################################
def augment(image, boxes, class_ids, crop_w=100, crop_h=100):
    
    transformer = A.Compose([  
                                A.Blur(blur_limit=7, always_apply=False, p=0.8),
                                A.RandomRotate90 (p=0.8),
                                A.RandomCrop (crop_h, crop_w, always_apply=False, p=0.2),
                                A.VerticalFlip (p=0.7),
                                A.HorizontalFlip (p=0.5)
                            ],
                          bbox_params=A.BboxParams(format='yolo', min_visibility=0.20, label_fields=['category_ids']))
    transformed = transformer(image=image, bboxes=boxes, category_ids=class_ids)
    aug_image = transformed['image']
    aug_boxes = list(map(list, transformed['bboxes']))
    aug_labels = [[transformed['category_ids'][idx]] + box for idx, box in enumerate(aug_boxes)]
    
    return aug_image, aug_labels

# run the augmentation pipeline ########################################################
def apply_augmentation(root_path, categories):
    for cat, num_augs in categories.items():
        
        img_src = os.path.join(root_path, "images")
        labels_src = os.path.join(root_path, "labels")
        
        img_dest = os.path.join(root_path, "aug images")
        labels_dest = os.path.join(root_path, "aug labels")
        
        
        if not os.path.exists(img_dest):
            os.makedirs(img_dest)
        if not os.path.exists(labels_dest):
            os.makedirs(labels_dest)
        
        
        names = [name.split(".")[0] for name in os.listdir(labels_src)]
        lbls_pths = [os.path.join(labels_src, name+".txt") for name in names]
        imgs_pths = [os.path.join(img_src, name+".jpg") for name in names]
        data = list(zip(imgs_pths, lbls_pths))
        
        for data_idx, data_file in enumerate(data):
            image, img_height, img_width, ratio = img_loader(data_file[0])
            box_coords, category_ids = load_yolo_annotations(data_file[1])
            
            for i in range(num_augs):
                aug_image, aug_labels = augment(image, box_coords, category_ids, crop_w=int(img_width/2), crop_h=int(img_height/2))
                
                aug_img_path = os.path.join(img_dest, f"{names[data_idx]}_aug_{i+1}.jpg")
                aug_lbl_path = os.path.join(labels_dest, f"{names[data_idx]}_aug_{i+1}.txt")
                
                aug_image = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                
                cv2.imwrite(aug_img_path, aug_image)       
                
                with open(aug_lbl_path, "w") as f:
                    for line in aug_labels:
                        line = list(map(str, line))
                        line.append('\n')
                        line_txt = " ".join(line)
                        f.write(line_txt)

# split the data for train, validation and test ########################################################
def split_data(dest_path, imgs_dir, labels_dir):
    
    train_dir = os.path.join(dest_path, "train")
    val_dir = os.path.join(dest_path, "val")
    test_dir = os.path.join(dest_path, "test")
    
    file_names = [file_name.split(".")[0] for file_name in os.listdir(imgs_dir)]
    total_num = len(file_names)
    
    train_num = int(total_num*0.6)
    val_num = test_num = int((total_num - train_num)/2)
    
    rand_idx = list(range(total_num)) 
    random.shuffle(rand_idx)
    
    train_idx = rand_idx[:train_num]
    val_idx = rand_idx[train_num:train_num+val_num]
    test_idx = rand_idx[train_num+val_num:]
    
    # train data
    train_imgs_path = os.path.join(train_dir, "images")
    train_labels_path = os.path.join(train_dir, "labels")
    
    if not os.path.isdir(train_imgs_path):
        os.makedirs(train_imgs_path)
    if not os.path.isdir(train_labels_path):
        os.makedirs(train_labels_path)
    
    for idx in train_idx:
        img_path, label_path = os.path.join(imgs_dir, file_names[idx]+".jpg"), os.path.join(labels_dir, file_names[idx]+".txt")
        img_dest, label_dest = os.path.join(train_imgs_path, file_names[idx]+".jpg"), os.path.join(train_labels_path, file_names[idx]+".txt")
        
        shutil.copy(img_path, img_dest)
        shutil.copy(label_path, label_dest)
        
    # val data
    val_imgs_path = os.path.join(val_dir, "images")
    val_labels_path = os.path.join(val_dir, "labels")
    
    if not os.path.isdir(val_imgs_path):
        os.makedirs(val_imgs_path)
    if not os.path.isdir(val_labels_path):
        os.makedirs(val_labels_path)
    
    for idx in val_idx:
        img_path, label_path = os.path.join(imgs_dir, file_names[idx]+".jpg"), os.path.join(labels_dir, file_names[idx]+".txt")
        img_dest, label_dest = os.path.join(val_imgs_path, file_names[idx]+".jpg"), os.path.join(val_labels_path, file_names[idx]+".txt")
        
        shutil.copy(img_path, img_dest)
        shutil.copy(label_path, label_dest)
        
    # test data
    test_imgs_path = os.path.join(test_dir, "images")
    test_labels_path = os.path.join(test_dir, "labels")
    
    if not os.path.isdir(test_imgs_path):
        os.makedirs(test_imgs_path)
    if not os.path.isdir(test_labels_path):
        os.makedirs(test_labels_path)
    
    for idx in test_idx:
        img_path, label_path = os.path.join(imgs_dir, file_names[idx]+".jpg"), os.path.join(labels_dir, file_names[idx]+".txt")
        img_dest, label_dest = os.path.join(test_imgs_path, file_names[idx]+".jpg"), os.path.join(test_labels_path, file_names[idx]+".txt")
        
        shutil.copy(img_path, img_dest)
        shutil.copy(label_path, label_dest)

    print(f"total number of images is {total_num}, training images {len(train_idx)}, validation images {len(val_idx)}, and test images {len(test_idx)}")                   

# loads the model, detects logo and then removes it
def remove_logo(model_path, img_src, img_dest, conf):

    model = YOLO(model_path)
    img = cv2.imread(img_src)
    
    result = model(img, conf=conf)[0]
    boxes = result.boxes.xyxy.cpu().numpy()


    x_min, y_min, x_max, y_max = int(boxes[0][0]), int(boxes[0][1]), int(boxes[0][2]), int(boxes[0][3])
    img[y_min:y_max, x_min:x_max] = img[y_max,x_max]

    cv2.imwrite(img_dest, img)
    
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)