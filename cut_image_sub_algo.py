import os
import numpy as np
import cv2
import random

def check(x, y, i, img_boxes, sub_img_shape):
    lt = sub_img_shape[0]
    rt = sub_img_shape[1]
    if (img_boxes[i][1] >= x and img_boxes[i][1] <= x + lt and img_boxes[i][2] >= y and img_boxes[i][2] <= y + rt
    and img_boxes[i][3] >= x and img_boxes[i][3] <= x + lt and img_boxes[i][4] >= y and img_boxes[i][4] <= y + rt):
        return True
    return False

def cover(x, y, box_selected, img_boxes, sub_img_shape):
    cnt = 0
    for i in range(len(box_selected)):
        if ((not box_selected[i]) and check(x, y, i, img_boxes, sub_img_shape)):
            cnt += 1
    return cnt

def box_select(x, y, box_selected, cropped_box_num, img_boxes, sub_img_shape):
    for i in range(len(box_selected)):
        if ((box_selected[i] == 0) and check(x, y, i, img_boxes, sub_img_shape)):
            box_selected[i] = cropped_box_num

def cut_img_to_subimg(height, width, img_boxes, sub_img_shape=(1600, 1600)):
    res = []
    lt = sub_img_shape[0]
    rt = sub_img_shape[1]
    img_boxes_sorted = sorted(img_boxes, key=lambda box: (box[1], box[2], box[3], box[4])) 
    box_selected = [0] * len(img_boxes_sorted)
    cropped_box_num = 0
    for i in range(len(img_boxes_sorted)):
        if (box_selected[i] == 0):
            xx1, yy1 = img_boxes_sorted[i][1], img_boxes_sorted[i][2] 
            score1 = cover(xx1, yy1, box_selected, img_boxes, sub_img_shape)
            xx2, yy2 = max(img_boxes_sorted[i][3] - lt, 0), img_boxes_sorted[i][4] 
            score2 = cover(xx2, yy2, box_selected, img_boxes, sub_img_shape)
            xx3, yy3 = img_boxes_sorted[i][1], max(img_boxes_sorted[i][4] - rt, 0) 
            score3 = cover(xx3, yy3, box_selected, img_boxes, sub_img_shape)
            xx4, yy4 = max(img_boxes_sorted[i][3] - lt, 0), max(img_boxes_sorted[i][4] - rt, 0) 
            score4 = cover(xx4, yy4, box_selected, img_boxes, sub_img_shape)
            max_score = max(score1, score2, score3, score4)
            if max_score == score1 and xx1 + lt <= width and yy1 + rt <= height:
                box_select(x=xx1, y=yy1, box_selected=box_selected, cropped_box_num=cropped_box_num, img_boxes=img_boxes, sub_img_shape=sub_img_shape)
                res.append((xx1, yy1))
            elif max_score == score2 and xx2 + lt <= width and yy2 + rt <= height:
                box_select(xx2, yy2, box_selected, cropped_box_num, img_boxes, sub_img_shape)
                res.append((xx2, yy2))
            elif max_score == score3 and xx3 + lt <= width and yy3 + rt <= height:
                box_select(xx3, yy3, box_selected, cropped_box_num, img_boxes, sub_img_shape)
                res.append((xx3, yy3))
            else:  
                box_select(xx4, yy4, box_selected, cropped_box_num, img_boxes, sub_img_shape)
                res.append((xx4, yy4))
            
            cropped_box_num += 1
    return res, box_selected
# cropped_image = image[cropped_y_min:cropped_y_max, cropped_x_min:cropped_x_max]
