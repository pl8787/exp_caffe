import os
import sys

tag = '_dlap'

solver_file_tpl = open('cifar10_quick_solver%s.prototxt.tpl' % tag).read()
gt_file_tpl = open('list_gt.txt.tpl').read()
trimap_file_tpl = open('list_trimap.txt.tpl').read()
data_file_tpl = open('list_data.txt.tpl').read()

pic_list = range(1, 28) #[1, 2, 4, 6, 8, 16, 18, 25, 26, 27]

for i in pic_list:
    open('cifar10_quick_solver%s.prototxt' % tag,'w').write(solver_file_tpl % i)
    open('list_gt%s.txt' % tag,'w').write(gt_file_tpl % i)
    open('list_trimap%s.txt' % tag,'w').write(trimap_file_tpl % i)
    open('list_data%s.txt' % tag,'w').write(data_file_tpl % i)
    
    print 'models%s/GT%02d' % (tag, i)
    
    os.mkdir('models%s/GT%02d' % (tag, i))
    
    os.system('train_quick%s.sh' % tag)
    
    os.system('python models%s/blob2matrix.py "models%s/GT%02d/cifar10_quick%s_iter_%%s_8_0.blob"' % (tag,tag,i,tag))