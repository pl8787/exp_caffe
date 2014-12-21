import caffe_pb2
import sys
import os

# Read sample submit file
f_name = []
sample_f = open('sampleSubmission.csv')
header = sample_f.readline()
for line in sample_f:
    f_name.append(line.split(',')[0])
sample_f.close()
print 'Read Sample File: OK!'

blob_file_tpl = 'bowl_test_result_iter_%s_p_%s'#sys.argv[1] #
out_file_tpl = 'result_iter_%s.csv'

def generate_submit_file(iter):
    out_file = open(out_file_tpl%iter, 'w')
    out_file.write(header)
    f_idx = 0
    line_idx = 0
    while os.path.isfile(blob_file_tpl % (iter, f_idx)):
        blob_file = blob_file_tpl % (iter, f_idx)
        blob = caffe_pb2.BlobProto()
        f = open(blob_file,'rb')
        blob.ParseFromString(f.read())
        f.close()
        print 'blob: %sx%sx%sx%s' % (blob.num, blob.channels, blob.height, blob.width)
        
        for i in range(blob.num):
            print >>out_file, '%s,'%f_name[line_idx] + ','.join( map(str, blob.data[i*121:(i+1)*121]) )
            line_idx += 1
        f_idx += 1
    out_file.close()
    
generate_submit_file(0)
        
    