import re
from glob import glob

train_loss_pattern = r'.+? Iteration (\d+), loss = (\d.\d+)'
test_loss_pattern = r'.+? Test score #0: (\d.\d+)'
test_acc_pattern = r'.+? Test score #1: (\d.\d+)'

files = glob("*.txt")

for f in files:
    out1 = open(f+'1.txt', 'w')
    out2 = open(f+'2.txt', 'w')
    out3 = open(f+'3.txt', 'w')

    train_loss = []
    test_loss = []
    test_acc = []
    for line in open(f):
        m1 = re.match(train_loss_pattern, line)
        m2 = re.match(test_loss_pattern, line)
        m3 = re.match(test_acc_pattern, line)
        
        if m1:
            train_loss.append( (m1.group(1), m1.group(2)) )
        if m2:
            test_loss.append( m2.group(1) )
        if m3:  
            test_acc.append( m3.group(1) )
        
    for x,y in train_loss:
        if int(x)%100 == 0:
            print >>out1, y#'%s\t%s'%(x, y)
        
    for x in test_loss:
        print >>out2, x
        
    for x in test_acc:
        print >>out3, x
        
    out1.close()
    out2.close()
    out3.close()