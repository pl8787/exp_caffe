import os
import sys

solver_file_tpl = open('cifar10_quick_solver_wide.prototxt.tpl').read()
gt_file_tpl = open('list_gt.txt.tpl').read()
trimap_file_tpl = open('list_trimap.txt.tpl').read()
data_file_tpl = open('list_data.txt.tpl').read()

for i in range(1, 28):
    open('cifar10_quick_solver_wide.prototxt','w').write(solver_file_tpl % i)
    open('list_gt.txt','w').write(gt_file_tpl % i)
    open('list_trimap.txt','w').write(trimap_file_tpl % i)
    open('list_data.txt','w').write(data_file_tpl % i)
    
    print 'models/GT%02d' % i
    
    os.mkdir('models/GT%02d' % i)
    
    os.system('train_quick_wide.sh')
    
    os.system('python models/blob2matrix.py "models/GT%02d/cifar10_quick_wide_iter_%%s_17_0.blob"' % i)