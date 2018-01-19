# Run the script with anaconda-python
# $ /home/<path to anaconda directory>/anaconda/bin/python LmdbClassification.py
import sys
import numpy as np
import lmdb
import caffe
from collections import defaultdict
caffe.set_mode_gpu()

# Modify the paths given below
deploy_prototxt_file_path = '/home/<username>/caffe/examples/cifar10/cifar10_deploy.prototxt' # Network definition file
caffe_model_file_path = '/home/<username>/caffe/examples/cifar10/cifar10_iter_5000.caffemodel' # Trained Caffe model file
test_lmdb_path = '/home/<username>/caffe/examples/cifar10/cifar10_test_lmdb/' # Test LMDB database path
mean_file_binaryproto = '/home/<username>/caffe/examples/cifar10/mean.binaryproto' # Mean image file

# Extract mean from the mean image file
mean_blobproto_new = caffe.proto.caffe_pb2.BlobProto()
f = open(mean_file_binaryproto, 'rb')
mean_blobproto_new.ParseFromString(f.read())
mean_image = caffe.io.blobproto_to_array(mean_blobproto_new)
f.close()

# CNN reconstruction and loading the trained weights
net = caffe.Net(deploy_prototxt_file_path, caffe_model_file_path, caffe.TEST)

count = 0
correct = 0
matrix = defaultdict(int) # (real,pred) -> int
labels_set = set()

lmdb_env = lmdb.open(test_lmdb_path)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()

for key, value in lmdb_cursor:
    datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        label = int(datum.label)
        image = caffe.io.datum_to_array(datum)
        image = image.astype(np.uint8)
    out = net.forward_all(data=np.asarray([image]) - mean_image)
    plabel = int(out['prob'][0].argmax(axis=0))
    count += 1
    iscorrect = label == plabel
    correct += (1 if iscorrect else 0)
    matrix[(label, plabel)] += 1
    labels_set.update([label, plabel])

    if not iscorrect:
            print("\rError: key = %s, expected %i but predicted %i" % (key, label, plabel))
        sys.stdout.write("\rAccuracy: %.1f%%" % (100.*correct/count))
        sys.stdout.flush()

print("\n" + str(correct) + " out of " + str(count) + " were classified correctly")
print ""
print "Confusion matrix:"
print "(r , p) | count"
for l in labels_set:
    for pl in labels_set:
        print "(%i , %i) | %i" % (l, pl, matrix[(l,pl)])
