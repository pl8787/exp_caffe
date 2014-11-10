import cv2
import numpy as np  
import caffe_pb2
import sys

blob_file_tpl = sys.argv[1] #'cifar10_quick_wide_iter_%s_17_0.blob'

def show_alpha(iter):
    blob_file = blob_file_tpl % iter
    blob = caffe_pb2.BlobProto()
    f = open(blob_file,'rb')
    blob.ParseFromString(f.read())
    f.close()

    print 'blob: %sx%sx%sx%s' % (blob.num, blob.channels, blob.height, blob.width)

    img = np.zeros((blob.height, blob.width), dtype=np.uint8)

    for idx, p in enumerate(blob.data):
        img[idx / blob.width, idx % blob.width] = np.uint8((p + 1.0) * 255)
        
    # cv2.imshow("alpha", img)
    cv2.imwrite(blob_file+".bmp", img)
    # cv2.waitKey(0)
    
for i in range(50, 101, 50):
    show_alpha(i)