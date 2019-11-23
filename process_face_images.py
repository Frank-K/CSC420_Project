import numpy as np
import os
import cv2
import random

def get_training_data(path):
    '''Returns two lists: a list of the names of the image files and a list of the bounding boxes'''
    #Open file
    f = open(path,'r') 
    file = f.readlines()
    name = []
    box = []
    count = 0
    for i in file: 
        #Grab the files with only 1 face
        if i == '1\n':
            filename = file[count-1]
            bb = file[count+1]
        
            #Get rid of the /n
            filename = filename[:-1]
            #Convert into array of integer coordinates
            num = bb.split(' ')
            num = num[:4]
            num = [int(i) for i in num]
            
            #Save into array
            name.append(filename)
            box.append(num)
        count = count+1
    return name, box

def create_scales(box):
    '''Create a list of scales that result in the face being 7,8,9,10,11,and 12 pixels big'''
    scale = np.empty([6])
    facesize = [12,11,10,9,8,7]
    width = box[2]
    height = box[3]
    larger = max(height, width)
    scale = [larger/i for i in facesize]
    return scale

def scale_image(image,scale):
    '''Scales image given the scale'''
    height, width, _ = image.shape
    width_scaled = int(np.ceil(width / scale))
    height_scaled = int(np.ceil(height / scale))
    im_data = cv2.resize(image, (width_scaled, height_scaled), interpolation=cv2.INTER_AREA)
    return im_data

def  scale_box (box,scale):
    '''Scales bounding box given the scale'''
    box = [int(i/scale) for i in box]
    return box

def draw_on_face(scaled_box, scaled_image):
    x, y, width, height = scaled_box
    x2, y2 = x + width, y + height
    # draw a rectangle over the pixels
    cv2.rectangle(scaled_image, (x, y), (x2, y2), (0,0,255), 1)

    # show the image
    cv2.imshow('face detection', scaled_image)
    # keep the window open until we press a key
    cv2.waitKey(0)
    # close the window
    cv2.destroyAllWindows()

# works for x and y
# make sure to use width for x, and height for y
def generate_offsets(original_coordinate, bounding_box_param, image_end):
    difference = 12 - bounding_box_param
    x_crops_start = []
    x_crops_end = []
    new_box_start = []
    for i in range(difference +  1):
        x_crop_start = original_coordinate - difference + i
        x_crops_end = original_coordinate + bounding_box_param + i
        if (x_crop_start >= 0 and x_crops_end <= image_end): # edge case
            x_crops_start.append(x_crop_start)
            x_crops_end.append()

            # new bounding box coordinates are on a 0 to 1 scale
            new_x = (original_coordinate - x_crop_start) / 12
            new_box_start.append(new_x)
    
    return x_crops_start, x_crops_end, new_box_start

def get_original_parameters(new_box):
    original_box = []
    for i in range(len(new_box)):
        original_box.append(int(new_box[i] * 12))
    
    return original_box

def main():
    
    imagepath = '../WIDER_train/images/'
    path = '../wider_face_split/wider_face_train_bbx_gt.txt'
    name, box = get_training_data(path)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    # print(dir_path)

    for i in range(len(name)):
        filename = name[i]
        bounding_box = box[i]
        
        # print(filename, bounding_box)

        name_without_extension = filename.split('.')[0]
        # print(name_without_extension)

        directory = name_without_extension.split('/')[0]
        # print("Directory:" + directory)
        if not os.path.exists('FaceTracker/train/' + directory):
            os.makedirs('FaceTracker/train/' + directory)

        scales = create_scales(bounding_box)
        image = cv2.imread(imagepath + filename)
        # print(image.shape)

        for i in range(len(scales)):
            scale = scales[i]
            scaled_image = scale_image(image, scale)
            scaled_box = scale_box(bounding_box, scale)
            # print(scaled_image.shape, scaled_box)
            # draw_on_face(scaled_box, scaled_image)

            # now, make as many 12x12 crops of faces as we can
            # say we have a 9x12 bounding box at top-left coordinates (30, 40)
            # Then we have:
            # 1. (30 - (12 -9), 40) to (30 + 9, 40)
            # 2. (30 - (12 - 10), 40) to (30 + (9 + 1), 40)
            # 3. (30 - (12 - 11), 40) to (30 + (9 + 2), 40)
            # 4. (30 - (12 - 12), 40) to (30 + (9 + 3), 40)

            # Thus, we have 4 images
            x, y, width, height = scaled_box

            x_difference = 12 - width
            y_difference = 12 - height
            # print(x_difference, y_difference)

            y_end = scaled_image.shape[0]
            x_end = scaled_image.shape[1]

            x_crops_start, x_crops_end, x_new_box_start = generate_offsets(x, width, x_end)
            y_crops_start, y_crops_end, y_new_box_start = generate_offsets(y, height, y_end)
            
            # width and height are simply divide by 12 and identical amongst these cropped images
            new_width = width / 12
            new_height = height / 12
            
            print(x_crops_start, x_crops_end, x_new_box_start)
            print(y_crops_start, y_crops_end, y_new_box_start)

            # now we need to crop all combinations of the x and y!
            for j in range(len(x_crops_start)):
                for k in range(len(y_crops_start)):
                    crop = scaled_image[y_crops_start[k]:y_crops_end[k], x_crops_start[j]:x_crops_end[j]]
                    # new_box = [x_new_box_start[j], y_new_box_start[k], new_width, new_height]
                    new_box = [x_new_box_start[j], y_new_box_start[k], new_width, new_height]

                    # original_box = get_original_parameters(new_box)
                    # print(new_box)
                    # draw_on_face(original_box, crop)

                    
                    # cv2.imshow("cropped", crop)
                    # cv2.waitKey(0)

                    # save crop image
                    # new_path = os.path.join(dir_path, 'FaceTracker', 'train', name_without_extension + 'crop_' + str(j) + '_' + str(k) + '.jpg')
                    new_file_name = name_without_extension + 'scale_' + str(scale) + '_crop_' + str(j) + '_' + str(k) + '.jpg'
                    new_path = './FaceTracker/train/'
                    
                    # print(new_file_name)
                    # print(crop.shape)
                    if not cv2.imwrite(new_path + new_file_name, crop):
                        raise Exception("Could not write image")
                    # write box_coordinates into txt file
                    with open(new_path + "bounding_box.txt", "a") as myfile:
                        myfile.write(new_file_name + '\n')
                        seperator = ' '
                        box_text = seperator.join("%10.3f" % x for x in new_box)
                        myfile.write(box_text + '\n')
        #     break
        # break

    return

def gen_neg_training(name, bbox, count):
    '''Generates 12x12 pixel images with no faces'''
    image = cv2.imread("../WIDER_train/images/" + name)
    scales=create_scales(bbox)
    print(scales)
    bbox = [bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]]
    a = name.split('/')
    b = a[1][:-4]
    
    #For each scale:
    for j in range(4):
        img=scale_image(image,scales[j])
        box = scale_box(bbox,scales[j])
        height, width, _ = img.shape
        i=0
        y1=0
        if width<=13 or height<=13:
            i=16
        while i<15:
            x=random.randint(0,width-13)
            y=random.randint(0,height-13)
            if y1>100: #If something went wrong and it keeps on falling into "continue", just break out of the loop
                i=16
                break
            if [x,y]>[box[0]-12,box[1]-12] and [x,y]<[box[2],box[3]]: #If the 12x12 box overlaps with the bounding box, draw another box
                y1=y1+1
                continue
            else: #If it doesn't overlap, crop the image.
                crop_img = img[y:y+12,x:x+12]
                cv2.imwrite(f'./FaceTracker/neg_train/{count}.bmp',crop_img)
                count = count+1
                i=i+1
    print("count",count)
    return count

def main2():
    imagepath = '../WIDER_train/images/'
    path = '../wider_face_split/wider_face_train_bbx_gt.txt'
    name, box = get_training_data(path)

    count = 0

    for i in range(len(name)):
        filename = name[i]
        bounding_box = box[i]

        count = gen_neg_training(filename, bounding_box, count)
        if count > 4214:
            return
        

if __name__ == '__main__':
    main2()