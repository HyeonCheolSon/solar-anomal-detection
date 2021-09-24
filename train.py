from __future__ import print_function
import argparse
import cv2
import numpy as np
import os

def mouse_event(event, x, y, flags, param):
    global pointlist
    
    if event == cv2.EVENT_FLAG_LBUTTON:    
        pointlist.append([x, y])
        print("point: " + str(x) + ", " + str(y))

def imwrite(filename, img, params=None): 
    try: 
        ext = os.path.splitext(filename)[1] 
        result, n = cv2.imencode(ext, img, params) 
        if result: 
            with open(filename, mode='w+b') as f: 
                n.tofile(f) 
            return True 
        else: 
            return False 
    except Exception as e: 
        print(e) 
        return False


if __name__ == "__main__" :

    # ap = argparse.ArgumentParser()
    # ap.add_argument('-i', '--image', required = True, help = 'Path to the input image')
    # args = vars(ap.parse_args())
    # filename = args['image']

    path_dir = 'C:\\Users\\손현철\\Desktop\\hc\\사업\\태양광\\패턴별 사진\\컬러'
    # path_dir = 'C:/Users/손현철/Desktop/hc/사업/태양광/패턴별 사진/컬러'
    output_dir = 'C:\\Users\\손현철\\Desktop\\hc\\사업\\태양광\\데이터만들기'
    
    folder_list = os.listdir(path_dir)
    
    for folder in folder_list:
        file_list = os.listdir(path_dir + "\\" + folder)
        # print(folder, file_list)

        for filename in file_list:
            # filename = path_dir + "/" + folder + '/' + filename
            filename = path_dir + "\\" + folder + '\\' + filename
            print(filename)
            # image = cv2.imread(filename)
            img_array = np.fromfile(filename, np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img = image.copy()
            pointlist = []
            cv2.imshow("draw", img)
            cv2.setMouseCallback("draw", mouse_event, img)
            cv2.waitKey()

            print(pointlist)
            if len(pointlist) == 0:
                continue

            # img = image.copy()
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # img = cv2.Canny(img, 40, 200)

            # (_, contours, _) = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            # for i in range(0,5):
            #     cntr = sorted(contours, key=cv2.contourArea, reverse=True)[i]
            #     epsilon = 0.1*cv2.arcLength(cntr, True)
            #     approx = cv2.approxPolyDP(cntr, epsilon, True)
            #     if(len(approx) == 4):
            #         cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
            #         break


            pts1 = np.float32(pointlist) # left-up - left-bottom - right-up - right-bottm
        #    pts1 = np.float32([[115, 173], [252, 928], [545, 91], [705, 827]]) # receipt.jpg
        #    pts1 = np.float32([[29, 132], [51, 978], [574, 65], [742, 892]]) # document.jpg
        #    pts1 = np.float32([[405, 192], [0, 631], [744, 409], [379, 931]]) # document2.jpg

            pts2 = np.float32([[0, 0], [0, 128], [128, 0], [128, 128]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            img = cv2.warpPerspective(image, matrix, (128, 128))
            # img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
            imwrite(output_dir + '\\컬러\\' + folder + '\\' + filename.split('\\')[-1].split('.')[-2] + "_data.jpg", img)
            print('saved: ' + output_dir + '\\컬러\\' + folder + '\\' + filename.split('\\')[-1].split('.')[-2] + "_data.jpg")

            # cv2.imwrite(output_dir + '\\컬러\\' + folder + '\\' + filename.split('\\')[-1].split('.')[-2] + "_data.jpg", img)
            # print('saved: ' + output_dir + '\\컬러\\' + folder + '\\' + filename.split('\\')[-1].split('.')[-2] + "_data.jpg")



