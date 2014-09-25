import caffe_pb2

net_file = 'cifar10_quick_co_iter_500'

net = caffe_pb2.NetParameter()
f = open(net_file,'rb')
net.ParseFromString(f.read())
f.close()
blob = net.layers[1].blobs[2]
data = blob.data

print blob.channels
print blob.num
print blob.width
print blob.height

p = 0

for f in range(blob.channels):
    print 'filter_%d.txt'%f
    out_f = open('filter_%d.txt'%f,'w')
    for w in range(blob.width):
        print >>out_f, '\t'.join(map(str, data[p:p+blob.height]))
        p += blob.height
    out_f.close()
    