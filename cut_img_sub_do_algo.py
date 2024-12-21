from cut_image_sub_algo import cut_img_to_subimg

import os
import numpy as np
import cv2

tot_cropped_num = 0
# 子图尺寸基底值D 
# the base value D of sub-image size
sub_img_size_base = 32 

def parse_label_box(label_file, img_width, img_height):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    boxes = []
    for line in lines:
        parts = line.strip().split()  
        class_id, x_center, y_center, width_ratio, height_ratio = map(float, parts)  
        class_id = int(class_id)
        x_pos, y_pos = x_center * img_width, y_center * img_height
        width, height = width_ratio * img_width, height_ratio * img_height
        x_min, y_min = x_pos - width // 2, y_pos - height // 2
        x_max, y_max = x_pos + width // 2, y_pos + height // 2
        boxes.append((class_id, x_min, y_min, x_max, y_max))
    
    return boxes

def images_parse(image_path, base_name, txt_path, img_output_path, txt_output_path, target_size=(1600, 1600)):
    # 读取图像和标注文件  
    # read image and label file
    image = cv2.imdecode(np.fromfile(image_path + '\\' + base_name + '.jpg', dtype=np.uint8), cv2.IMREAD_COLOR) 
    txt_file_dir = txt_path + base_name + '.txt'  

    
    image_boxes = parse_label_box(txt_file_dir, image.shape[1], image.shape[0])

    short_edge = min(image.shape[0], image.shape[1])
    # 计算short_edge > n*D的最大n
    # calculate the maximum n of short_edge > n*D
    n = short_edge // sub_img_size_base
    
    for j in range(n):
        target_size = (sub_img_size_base * (j + 1), sub_img_size_base * (j + 1))

        cropped_boxes, box_selected = cut_img_to_subimg(
            height=image.shape[0],
            width=image.shape[1],
            img_boxes=image_boxes,
            sub_img_shape=target_size
        )
        # 使用列表来累积修改后的标注信息
        # use list to accumulate modified label information
        cropped_labels = []
        for i in range(len(cropped_boxes)):
            raw_img = image.copy()
            x, y = cropped_boxes[i]
            y_max = y + target_size[1]
            x_max = x + target_size[0]
            # 切割图像
            # crop image
            cropped_image = raw_img[int(y):int(y_max), int(x):int(x_max)]
            # 获取当前cropped box内的标注框的下标
            # get the index of the annotation box in the current cropped box
            # indices = [i for i, value in enumerate(box_selected) if value == i] 
            indices = []
            for annotation_id in range(len(box_selected)):
                if box_selected[annotation_id] == i:
                    indices.append(annotation_id)
            print(f' No.{i} part of image [{base_name}] cropped from ({x}, {y}), \n to ({x_max}, {y_max})')
            # print(box_selected)
            # 更新标注信息
            # update label information
            cropped_labels = []  
            label_cnt = 0
            for indice in indices:
                box = image_boxes[indice]
                cls_id, x_min, y_min, x_max, y_max = box
                after_x_min = max(0, x_min - x) # 限制范围
                after_y_min = max(0, y_min - y) # 限制范围
                after_x_max = min(target_size[0], x_max - x) # 限制范围  
                after_y_max = min(target_size[1], y_max - y) # 限制范围

                # 计算裁剪后的YOLO格式标注信息
                # calculate the YOLO format label information after cropping
                after_x_center = ((after_x_min + after_x_max) / 2) / target_size[0]
                after_y_center = ((after_y_min + after_y_max) / 2) / target_size[1]
                after_width = (after_x_max - after_x_min) / target_size[0]
                after_height = (after_y_max - after_y_min) / target_size[1]

                cropped_labels.append((cls_id, after_x_center, after_y_center, after_width, after_height))
                print(f'    No.{label_cnt} label box transformed to ({after_x_min}, {after_y_min})\n ({after_x_max}, {after_y_max})')
                label_cnt += 1

            # 写入数据
            # write data
            # 保存裁剪后的图像和标注信息
            # save cropped image and label information
            try:
                cv2.imencode('.jpg', cropped_image)[1].tofile(img_output_path + '\\' + base_name + f'_cropped_part_{str(i)}.jpg')
            except Exception as e:
                print(f"Error saving image: {e}")
            
            with open(txt_output_path + base_name + f'_cropped_part_{str(i)}.txt', 'w') as f:
                for label in cropped_labels:
                    f.write(f"{label[0]} {label[1]} {label[2]} {label[3]} {label[4]}\n")
    
        global tot_cropped_num
        tot_cropped_num += len(cropped_boxes)
    
    print(f'======= Image {base_name} cut into {len(cropped_boxes)} parts =======')

def main():
    image_path = "your_origin_image_path"
    txt_path = "your_origin_label_path"
    img_output_path = "your_cropped_image_path"
    txt_output_path = "your_cropped_label_path"

    # target_size = (1600, 1600)
    target_size = (960, 960)
    image_files = os.listdir(image_path)
    for img_file in image_files:
        if img_file.endswith('.jpg'):
            base_name = os.path.splitext(img_file)[0]
            images_parse(image_path, base_name, txt_path, img_output_path, txt_output_path, target_size)
    
    print()
    print(f'Total image number: {len(image_files)}, Total cropped number: {tot_cropped_num}')

if __name__ == '__main__':
    main()
