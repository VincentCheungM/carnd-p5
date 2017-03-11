import cv2
import numpy as np
import glob
import pickle

def calibrate(CONRX, CONRY, imgDir = 'camera_cal/calibration*.jpg', paraFile='camera.p', drawChess=False): 
    imgs = glob.glob(imgDir)
    i = 0

    #init obj & img points
    objp = np.zeros((CONRX*CONRY,3), np.float32)
    objp[:,:2] = np.mgrid[0:CONRX, 0:CONRY].T.reshape(-1,2)        
    objPoints = []
    imgPoints = []

        #convert to gray scale and find corners
        #img = cv2.imread(imgName)
    for idx, fname in enumerate(imgs):
        print ('calibrating '+fname)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (CONRX,CONRY), None)
        
        if ret == True:
            objPoints.append(objp)
            imgPoints.append(corners)
            #if need to draw chessboard
            if drawChess == True:
                cv2.drawChessboardCorners(img, (CONRX,CONRY), corners, ret)
                cv2.imwrite('./undisort/chessBoard/'+fname,img)
            #calibrate matrix
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], None, None)
                    #dump to file
            distPkl = {}
            distPkl['mtx'] = mtx
            distPkl['dist'] = dist
            pickle.dump(distPkl, open(paraFile,'wb'))
        else:
            print ("Not find corners on img {}".format(fname))
    return (mtx, dist)
  


def undisort(img, mtx, dist):
    try:
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        return dst
    except Exception as e: 
        print(str(e))    

def loadCalMat(paraFile='camera.p'):
    try:
        return pickle.load(open(paraFile,mode='rb'))
    except Exception as e: 
        print(str(e)) 

def calPipeLine():
    cornx = 9
    corny = 6
    imgDir = 'camera_cal/calibration*.jpg'
    calibrate(cornx, corny, imgDir, paraFile='camera.p', drawChess=True)
    imgs = glob.glob(imgDir)
    distPkl = loadCalMat()
    for idx, fname in enumerate(imgs):

        img = cv2.imread(fname)
        uds = undisort(img,distPkl['mtx'], distPkl['dist'])
        cv2.imwrite('./undisort/'+fname,uds)
        print ('undisorting '+fname)
    print ('Process end')

if __name__ == '__main__':
    calPipeLine()