import cv2
import numpy as np



keep_original= True
def getEyeArea(img):
    """gets input of face and returns list of detected eye areas"""
    ret = []
    eye_cascade = cv2.CascadeClassifier(r'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(img, 1.3, 5)
    return eyes

def erase_except_eyes(img, eyes, margin=0, backgraound_weight=0):
    #blank_image = np.zeros((256, 256, 3), np.uint8)

    backgraound_img= (img.astype(float)*backgraound_weight).astype(int)

    for (ex, ey, ew, eh) in eyes:
        # add margins around eye area
        begin_x= max(ex-margin,0)
        end_x= min(ex+ew+margin, 255)
        begin_y= max(ey-margin,0)
        end_y = min(ey +eh+ margin, 255)
        #copy the eye area to the blank image
        backgraound_img[ begin_y:end_y,begin_x:end_x] = img[ begin_y:end_y,begin_x:end_x]
    return backgraound_img



# def erase_except_eyes(img, eyes, margin=0):
#     blank_image = np.zeros((256, 256, 3), np.uint8)
#     img= (img.astype(float)*0.5).astype(int)
#
#     for (ex, ey, ew, eh) in eyes:
#         # add margins around eye area
#         begin_x= max(ex-margin,0)
#         end_x= min(ex+ew+margin, 255)
#         begin_y= max(ey-margin,0)
#         end_y = min(ey +eh+ margin, 255)
#         #copy the eye area to the blank image
#         blank_image[ begin_y:end_y,begin_x:end_x] = img[ begin_y:end_y,begin_x:end_x]
#     return blank_image

def erase_bottom_face(img):
    blank_image = np.zeros((256, 256, 3), np.uint8)
    for (ex, ey, ew, eh) in eyes:
        blank_image[0:128] = img[0:128]
    return blank_image




def test(path):
    """1 is normal color, 0 is grayscale, -1 is image with an alpha channel (transparency)"""
    img = cv2.imread(path)
    eyes = getEyeArea(img)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,127,255),5)
        crop_img = img[ex:ex+ew, ey:ey+eh]
        cv2.imshow("cropped", crop_img)

    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#test(r"C:\Users\askle\PycharmProjects\deep learniing course\Age-Gender-Pred-master\pics\train\13_0_ColleenCorby.jpg")


import os
dir_path = os.path.dirname(os.path.realpath(__file__))


for phase in ['train', 'val']: # for images of both phases (train and validation)
    for filename in os.listdir(dir_path+"/pics/"+phase): # iterate on all images inside the folder
        img = cv2.imread(dir_path+"/pics/"+phase+'/'+filename)
        eyes = getEyeArea(img)
        if(len(eyes)):
            if(not keep_original):
                img=erase_except_eyes(img,eyes,10,0.6)
                cv2.imwrite(dir_path+r"/pics/_"+phase+'/'+filename, img)
            else:
                cv2.imwrite(dir_path + r"/pics/o_" + phase + '/' + filename, img)


