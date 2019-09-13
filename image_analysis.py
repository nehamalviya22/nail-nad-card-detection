import os
import label_detector
import myutils
data_path = myutils.DATA_PATH
import numpy as np
import requests
import json
import google_uploader
from firebase import firebase
import shutil
import traceback
import subprocess
import cv2

import time
import imutils
from imutils import contours
from imutils import perspective
from scipy.spatial import distance as dist
from PIL import  Image

template_width = 85.6 # physical width in mm

class ImageAnalyser(object):
    def __init__(self,i_id,i_url=None,i_file=None):

        self.i_id = i_id
        self.i_url = i_url
        self.img_file = i_file

        #self.a_upl = aws_uploader.AwsUploader()
        self.a_upl = google_uploader.GoogleUploader()
        
        self.a_upl.makeConnection()
        self.progress = 1


    def classify_scene(self):
        try:
            self.op_path = data_path + self.i_id+'_'+ str(int(time.time()))+'/'
            self.img_path = self.op_path+"/"+self.i_id+".jpg"

            if os.path.exists(self.op_path):
                shutil.rmtree(self.op_path)

            if not os.path.exists(self.op_path):
                os.makedirs(self.op_path)
                os.makedirs(self.op_path + "/image_data/")

            status = myutils.download_image(self.i_url,self.img_path)
            if not status:
                pass
            img = cv2.imread(self.img_path)
            img = imutils.rotate_bound(img, 90)
            cv2.imwrite(self.img_path, img)

            self.fb = firebase.FirebaseApplication('https://toch-vendor.firebaseio.com', None)


            self.labelFinder = label_detector.LabelFinder()
            label_all = self.labelFinder.find_label_by_darknet([self.img_path])

            pixelsPerMetric = self.get_ppm(self.img_path)
            for b_obj in label_all[0]:

                bbox = b_obj['box']
                width_in_pixel = bbox[2] - bbox[0]
                height_in_pixel = bbox[3] - bbox[1]
                b_obj['width_pixel'] = width_in_pixel
                b_obj['height_pixel'] = height_in_pixel
                b_obj['width_ph'] = width_in_pixel * pixelsPerMetric
                b_obj['height_ph'] = height_in_pixel * pixelsPerMetric
                cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),3)
                d = str(int(b_obj['height_ph']))
                e = str(int(b_obj['width_ph']))
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img,e+","+d+"mm",(bbox[0],bbox[1]), font, 2,(0,0,255),2,cv2.LINE_AA)
           # print("label_all",label_all)
            nail_data = self.create_data_for_api_nail(label_all)
            print (nail_data)
            cv2.imwrite(self.op_path + "detected_img.jpg", img)
            return nail_data

            #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            #cv2.resizeWindow('image', 600, 600)
            #cv2.imshow("image",img)
            #cv2.waitKey(0)


            #self.upload_gif_ons3(gif_mng.giflist)


            # gif_data = self.create_data_for_api_gif()
            #  self.post_data_on_api_gif(gif_data)

            # if os.path.exists(data_path + self.v_id):
            #     shutil.rmtree(data_path + self.v_id)
        except:
            traceback.print_exc()
            #self.post_data_on_api_gif('',status='failed')
            self.progress = 100

    def analyse_image_by_file(self):
        try:
            self.op_path = data_path + self.i_id+'_'+ str(int(time.time()))+'/'
            self.img_path = self.op_path+"/"+self.i_id+".jpg"

            if os.path.exists(self.op_path):
                shutil.rmtree(self.op_path)

            if not os.path.exists(self.op_path):
                os.makedirs(self.op_path)
                os.makedirs(self.op_path + "/image_data/")

            status = True # myutils.download_image(self.i_url,self.img_path)
            image = Image.open(self.img_file)
            image.save(self.img_path)

            if not status:
                pass


            img = cv2.imread(self.img_path)
            #img = imutils.rotate_bound(img, 90)
            cv2.imwrite(self.img_path, img)
            img_height,img_width,img_c = img.shape

            self.fb = firebase.FirebaseApplication('https://toch-vendor.firebaseio.com', None)


            self.labelFinder = label_detector.LabelFinder()
            label_all = self.labelFinder.find_label_by_darknet([self.img_path])

            pixelsPerMetric = self.get_ppm(self.img_path)
            for b_obj in label_all[0]:

                bbox = b_obj['box']
                width_in_pixel = bbox[2] - bbox[0]
                height_in_pixel = bbox[3] - bbox[1]
                b_obj['width_pixel'] = width_in_pixel
                b_obj['height_pixel'] = height_in_pixel
                try:
                    b_obj['width_ph'] = width_in_pixel * pixelsPerMetric
                    b_obj['height_ph'] = height_in_pixel * pixelsPerMetric
                except:
                    res = dict()
                    res['status'] = False
                    res['result'] = "credit card not found"
                    return res
                cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),3)
                d = str(int(b_obj['height_ph']))
                e = str(int(b_obj['width_ph']))
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img,e+","+d+"mm",(bbox[0],bbox[1]), font, 2,(0,0,255),2,cv2.LINE_AA)
           # print("label_all",label_all)
            nail_data = self.create_data_for_api_nail(label_all)
            print (nail_data)
            cv2.imwrite(self.op_path + "detected_img.jpg", img)
            res = dict()
            res['status'] = True
            res['result'] = nail_data
            return res

            #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            #cv2.resizeWindow('image', 600, 600)
            #cv2.imshow("image",img)
            #cv2.waitKey(0)


            #self.upload_gif_ons3(gif_mng.giflist)


            # gif_data = self.create_data_for_api_gif()
            #  self.post_data_on_api_gif(gif_data)

            # if os.path.exists(data_path + self.v_id):
            #     shutil.rmtree(data_path + self.v_id)
        except:
            traceback.print_exc()
            #self.post_data_on_api_gif('',status='failed')
            self.progress = 100

    def analyse_image_by_file_new(self):
        try:
            self.op_path = data_path + self.i_id+'_'+ str(int(time.time()))+'/'
            self.img_path = self.op_path+"/"+self.i_id+".jpg"

            if os.path.exists(self.op_path):
                shutil.rmtree(self.op_path)

            if not os.path.exists(self.op_path):
                os.makedirs(self.op_path)
                os.makedirs(self.op_path + "/image_data/")

            status = True # myutils.download_image(self.i_url,self.img_path)
            image = Image.open(self.img_file)
            image.save(self.img_path)

            if not status:
                pass


            img = cv2.imread(self.img_path)
            #img = imutils.rotate_bound(img, 90)
            cv2.imwrite(self.img_path, img)
            img_height,img_width,img_c = img.shape

            self.fb = firebase.FirebaseApplication('https://toch-vendor.firebaseio.com', None)


            self.labelFinder = label_detector.LabelFinder()
            label_all = self.labelFinder.find_label_by_darknet([self.img_path])

            for b_obj in label_all[0]:

                bbox = b_obj['box']
                label_i = b_obj['label']
                #width_in_pixel = bbox[2] - bbox[0]
                #height_in_pixel = bbox[3] - bbox[1]

                b_obj['x1'] = bbox[0]
                b_obj['y1'] = bbox[1]
                b_obj['x2'] = bbox[2]
                b_obj['y2'] = bbox[3]
                b_obj['label'] = label_i
                cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),3)

                #d = str(int(b_obj['height_pixel']))
               # e = str(int(b_obj['width_pixel']))
                #font = cv2.FONT_HERSHEY_SIMPLEX
                #cv2.putText(img,e+","+d+"mm",(bbox[0],bbox[1]), font, 2,(0,0,255),2,cv2.LINE_AA)
           # print("label_all",label_all)
            nail_data = self.create_data_for_api_nail(label_all)
            print (nail_data)
            cv2.imwrite(self.op_path + "detected_img.jpg", img)
            res = dict()
            res['status'] = True
            res['result'] = nail_data
            return res

            #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            #cv2.resizeWindow('image', 600, 600)
            #cv2.imshow("image",img)
            #cv2.waitKey(0)


            #self.upload_gif_ons3(gif_mng.giflist)


            # gif_data = self.create_data_for_api_gif()
            #  self.post_data_on_api_gif(gif_data)

            # if os.path.exists(data_path + self.v_id):
            #     shutil.rmtree(data_path + self.v_id)
        except:
            traceback.print_exc()
            #self.post_data_on_api_gif('',status='failed')
            self.progress = 100

    def midpoint(self,ptA, ptB):
        return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

    def get_ppm(self,img_path):
        image = cv2.imread(img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        edged = cv2.Canny(gray, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        (cnts, _) = contours.sort_contours(cnts)
        pixelsPerMetric = None
        for c in cnts:

            if cv2.contourArea(c) < 100:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.01 * peri, True)
            if len(approx) != 4:
                continue

            # compute the rotated bounding box of the contour
            orig = image.copy()
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)

            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 4)
            for (x, y) in box:
                cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)


            (tl, tr, br, bl) = box
            (tltrX, tltrY) = self.midpoint(tl, tr)
            (blbrX, blbrY) = self.midpoint(bl, br)
            (tlblX, tlblY) = self.midpoint(tl, bl)
            (trbrX, trbrY) = self.midpoint(tr, br)
            cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

            # draw lines between the midpoints
            cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                     (255, 0, 255), 2)
            cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                     (255, 0, 255), 2)
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            # if the pixels per metric has not been initialized, then
            # compute it as the ratio of pixels to supplied metric
            # (in this case, inches)
            if pixelsPerMetric is None:
                pixelsPerMetric = template_width / dB

            # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('image', 600, 600)
            # cv2.imshow("image",orig)
            # cv2.waitKey(0)

            return pixelsPerMetric

    def create_data_for_api_nail(self,label_all):
        data = []
        for obj_i in label_all[0]:
            print (obj_i)
            obj = {}
            obj['height_ph'] = int(obj_i['height_ph'])
            obj['width_ph'] = int(obj_i['width_ph'])
            '''
            obj['x1'] = int(obj_i['x1'])
            obj['y1'] = int(obj_i['y1'])
            obj['x2'] = int(obj_i['x2'])
            obj['y2'] = int(obj_i['y2'])
            obj['label'] = str(obj_i['label'])
            '''
            data.append(obj)

        return data



    def create_data_for_api_gif(self):
        data = []
        for scene_i in self.scene_in_video:
            obj = {}
            obj['scene_id'] = scene_i['scene_id']
            obj['s_ts'] = scene_i['s_ts']
            obj['e_ts'] = scene_i['e_ts']
            obj['gif'] = scene_i['gif']
            obj['scene_url'] = scene_i.get('scene_url_s3','')
            obj['label'] = ','.join(scene_i['label_all'])

            data.append(obj)

        return data

    def post_data_on_api_gif(self, gif_data,status=''):
        dataObj = {}

        dataObj['requestId'] = self.i_id
        dataObj['gifs'] = gif_data
        dataObj["job_id"] = self.job_id
        if status!='':
            dataObj['status'] = status


        v_domain = 'vendor.mytoch.com'

        # url = "http://139.59.18.180:8001/api/v1/videos/update/urls"
        url = "https://" + v_domain + "/api/v1/ai/request/edit"

        headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
        r = requests.post(url, data=json.dumps(dataObj), headers=headers, verify=False)
        print(dataObj)
        print("json post response {}".format(r.text))