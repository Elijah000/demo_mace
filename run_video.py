import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-Video') #日志記錄
logger.setLevel(logging.DEBUG)# 設置記錄器的級別
ch = logging.StreamHandler() #將日誌訊息送到設置的位置 
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s') #創建日誌的格式和時間格式
ch.setFormatter(formatter) #格式化
logger.addHandler(ch)

fps_time = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video') #宣告引數 可以透過前者 + 後者來輸入引數
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')#解析度
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model))) #將後兩者丟到前面作為debug
    w, h = model_wh(args.resolution) #將引數輸入的解析度丟到w,h兩變數裡
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h)) #tf-pose檔案的部分
    cap = cv2.VideoCapture(args.video) #抓引數的video

    if cap.isOpened() is False: #檢查是否成功初始化
        print("Error opening video stream or file")
    while cap.isOpened():
        ret_val, image = cap.read() #ret代表成功與否，image代表攝影機的單張畫面

        humans = e.inference(image) #tf-pose的部分
        if not args.showBG:  #沒成功，show default message
            image = np.zeros(image.shape) #
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False) #tf-pose的部分

        cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)#在圖像上繪製文字，fps計算公式
        cv2.imshow('tf-pose-estimation result', image) #show image
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows() #關窗
logger.debug('finished+') #結束訊息
