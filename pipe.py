import cv2
import numpy as np
import matplotlib.pyplot as plt

from calibrate import undisort, loadCalMat

from sklearn.externals import joblib
import pickle
from scipy.ndimage.measurements import label
from window import slide_window, search_windows, draw_boxes, add_heat, apply_threshold, draw_labeled_bboxes

def pipeline(img, diskPkl, svc, X_scaler, windows):
    #Calibrate camera & undistort img
    ## after calibratoin by running python calibrate.py
    #Extract mtx, dist from diskPkl
    mtx = diskPkl['mtx']
    dist = diskPkl['dist']
    undisImg = undisort(img, mtx, dist)
    heat = np.zeros_like(undisImg[:,:,0]).astype(np.float)


    #parameter for features
    cspace = 'YCrCb' #color space
    spatial_size = (32,32) #spatial_size
    hist_bins = 32 #histogram bins
    orient = 9 # hog orient
    pix_per_cell = 8 
    cell_per_block = 2
    hog_channel = 'ALL' #
    spatial_feat = True
    hist_feat = True
    hog_feat = True

    box_list = search_windows(undisImg, windows, svc, X_scaler, color_space=cspace, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    heat = add_heat(heat,box_list)
    heat = apply_threshold(heat,1)
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(undisImg), labels)
    #draw_img = draw_boxes(undisImg, box_list, color=(0, 0, 255), thick=6)
    return draw_img



if __name__ == "__main__":
    #load camera calibration matrix
    diskPkl = loadCalMat()

    #load svc
    svc = joblib.load('./svm_model/svc.pkl')
    X_scaler = pickle.load(open("./svm_model/X_scaler.pkl", "rb"))

    #for single image test
    """
    import glob
    test_imgs = glob.glob('./test_images/test*.jpg')

    for imgs in test_imgs:
        img = cv2.imread(imgs)
        windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[360, None], 
                    xy_window=(96, 96), xy_overlap=(0.5, 0.5))
        result = pipeline(img,diskPkl,svc,X_scaler, windows)
        cv2.imwrite('./output_images/'+imgs[-5:],result)
    """
    cap = cv2.VideoCapture('project_video.mp4')
    ret, frame = cap.read()
    windows = slide_window(frame, x_start_stop=[None, None], y_start_stop=[360, None], 
                    xy_window=(96, 96), xy_overlap=(0.5, 0.5))
    count = 0
    while frame is not None:
        result = pipeline(frame, diskPkl, svc, X_scaler, windows)
        ret, frame = cap.read()
        cv2.imwrite('./videos/'+str(count)+'.jpg', result)
        count += 1
        cv2.imshow('video',result)
        #out.write(result)
        #out.write(result)
        cv2.waitKey(1)
    cap.release()
    
    #cv2.destroyAllWindows()
    print ('done')