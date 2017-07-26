import os
import numpy as np
import cv2 as cv

data_dir_path = "./animeface-character-dataset/thumb/"
tmp = os.listdir(data_dir_path)
tmp=sorted([x for x in tmp if os.path.isdir(data_dir_path+x)])
dir_list = tmp

X_target=[]
for dir_name in dir_list:
    file_list = os.listdir(data_dir_path+dir_name)
    for file_name in file_list:
        if file_name.endswith('.png'):
            image_path=str(data_dir_path)+str(dir_name)+'/'+str(file_name)
            image = cv.imread(image_path)
            image = cv.resize(image, (32, 32))
            image = image.transpose(2,0,1)
            image = image/255.
            X_target.append(image)

anime_class=[]
count=0
for dir_name in dir_list:
    file_list = os.listdir(data_dir_path+dir_name)
    for file_name in file_list:
        if file_name.endswith('.png'):
            anime_class.append(count)
    count+=1

anime_arr2=np.array(anime_class)
np.save('anime_face_target.npy',anime_arr2)
anime_arr=np.array(X_target)
np.save('anime_face_data.npy',anime_arr)