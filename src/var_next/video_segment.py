import numpy as np
import cv2
import time
import pandas as pd
cap = cv2.VideoCapture("C:\\Users\\Alekche\\Documents\\UVA\\geo-of-data\\VAR-next\\full_log2_320.mp4")
fgbg = cv2.createBackgroundSubtractorMOG2()
length_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f'frames: {length_vid}')
OPENCV_OBJECT_TRACKERS = {
		#"csrt": cv2.TrackerCSRT_create,
		"kcf": cv2.TrackerKCF_create,
		#"boosting": cv2.TrackerBoosting_create,
		#"mil": cv2.TrackerMIL_create,
		#"tld": cv2.TrackerTLD_create,
		#"medianflow": cv2.TrackerMedianFlow_create,
		#"mosse": cv2.TrackerMOSSE_create
	}
	# grab the appropriate object tracker using our dictionary of
	# OpenCV object tracker objects
initBB=None

Tx = 203
Txx = Tx+39
Ty = 171
Tyy = Ty + 8
miss = 0
miss_dd = 0
frame_res = np.zeros([length_vid,4])
idx = 0
tbox = None
for _ in range(length_vid):
    ret, frame = cap.read()
    print(f'{(int(idx/length_vid)*100)}% complete')
    try:
        dd = tbox - frame[Ty:Tyy,Tx:Txx]
        print(np.mean(dd))
        if np.mean(dd) > 125:
            miss_dd+=1
            if miss_dd >3:
                initBB = None
                miss_dd = 0
        else:
            miss_dd = 0
    except:
        pass
    #fgmask = fgbg.apply(frame,learningRate=0.7)
    if initBB:
        (success, box) = tracker.update(frame)
            # check to see if the tracking was a success
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                (0, 255, 0), 2)
            last_box = np.array([x,y,w,h])
            frame_res[idx,:] = last_box
            #cv2.imshow("test",frame[y:(y+h_),x:(x+w_)])
        else:
            miss+=1
            if miss >5:
                initBB = None
            else:
                frame_res[idx,:] =  last_box
# res,thresh = cv2.threshold(fgmask,127,255,0)
    #kernel = np.ones((10,10),np.uint8)
    #dilation = cv2.dilate(fgmask,kernel,iterations = 1)
    #erosion = cv2.erode(fgmask,kernel,iterations = 1)
    #contours,hierarchy = cv2.findContours(fgmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    #or i in range(0, len(contours)):
    #    if (i % 1 == 0):
    #        cnt = contours[i]


    #        x,y,w,h = cv2.boundingRect(cnt)
    #        #cv2.drawContours(fgmask ,contours, -1, (255,255,0), 3)
    #        cv2.rectangle(fgmask,(x,y),(x+w,y+h),(255,0,0),2)
    key = cv2.waitKey(1) & 0xFF
	# if the 's' key is selected, we are going to "select" a bounding
	# box to track
    if key == ord("s") or initBB==None:
        initBB=[0]
		# select the bounding box of the object we want to track (make
		# sure you press ENTER or SPACE after selecting the ROI)
        while np.sum(initBB) == 0:
            initBB = cv2.selectROI("Frame", frame, fromCenter=False,showCrosshair=True)
        print(initBB)
		# start OpenCV object tracker using the supplied bounding box
		# coordinates, then start the FPS throughput estimator as well
        #import pdb; pdb.set_trace()
        tracker = OPENCV_OBJECT_TRACKERS['kcf']()
        tracker.init(frame, initBB)

        (x, y, w, h) = [int(v) for v in initBB]
        last_box = np.array([x,y,w,h])
        frame_res[idx,:] =  last_box
        #Select timestamp at bottom of image:
        tbox = frame[Ty:Tyy,Tx:Txx]
        miss_dd = 0
        miss = 0
    #cv2.rectangle(frame, (Tx, Ty), (Txx, Tyy),
    #        (255, 0, 0), 1)


    #cv2.imshow('frame',fgmask)
    f2 = cv2.resize(frame,(640,360))
    cv2.imshow("original",f2)
    #cv2.resizeWindow("original", 640) 
    idx+=1
    if key == ord('q'):
        break
df = pd.DataFrame(frame_res)
df.columns = ['x','y','dx','dy']
df.to_csv('ANNOTfull_log2_320.csv')
cap.release()
cv2.destroyAllWindows()