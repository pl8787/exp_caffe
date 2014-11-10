import cv2
import numpy as np 

gt_tpl = r"C:/pangl/Caffe/exp_caffe/examples/matting/data/gt_training_lowres/GT%02d.png"

in_tpl = r"C:/pangl/Caffe/exp_caffe/examples/matting/data/input_training_lowres/GT%02d.png"

trimap1_tpl = r"C:/pangl/Caffe/exp_caffe/examples/matting/data/trimap_training_lowres/Trimap2/GT%02d.png"

trimap2_tpl = r"C:/pangl/Caffe/exp_caffe/examples/matting/data/trimap_training_lowres/Trimap2/GT%02d.png"

out_t1_tpl = r"C:/pangl/Caffe/exp_caffe/examples/matting/data/gt_cnn_training_lowers/Trimap1/GT%02d.bmp"

out_t2_tpl = r"C:/pangl/Caffe/exp_caffe/examples/matting/data/gt_cnn_training_lowers/Trimap2/GT%02d.bmp"

for i in range(1,28):
    gt = gt_tpl % i
    print gt
    input = in_tpl % i
    trimap1 = trimap1_tpl % i
    trimap2 = trimap2_tpl % i
    out_trimap1 = out_t1_tpl % i
    out_trimap2 = out_t2_tpl % i
    
    img_gt = cv2.imread(gt, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    img_trimap1 = cv2.imread(trimap1, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    img_trimap2 = cv2.imread(trimap2, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    
    img_gt1 = np.zeros((img_gt.shape[0],img_gt.shape[1]), dtype=img_gt.dtype)  
    img_gt2 = np.zeros((img_gt.shape[0],img_gt.shape[1]), dtype=img_gt.dtype) 
    
    for x in range(img_gt.shape[0]):
        for y in range(img_gt.shape[1]):
            if img_trimap1[x, y]==128:
                img_gt1[x, y] = img_gt[x, y]
            else:
                img_gt1[x, y] = 128
            if img_trimap2[x, y]==128:
                img_gt2[x, y] = img_gt[x, y]
            else:
                img_gt2[x, y] = 128    
    
    cv2.imwrite(out_trimap1, img_gt1)
    cv2.imwrite(out_trimap2, img_gt2)
                
                
                