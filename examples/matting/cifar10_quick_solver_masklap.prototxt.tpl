# reduce the learning rate after 8 epochs (4000 iters) by a factor of 10

# The training protocol buffer definition
train_net: "cifar10_quick_train_masklap.prototxt"
# The testing protocol buffer definition
test_net: "cifar10_quick_test_masklap.prototxt"
# test_iter specifies how many forward passes the test should carry out.
# In the case of MNIST, we have test batch size 100 and 100 test iterations,
# covering the full 10,000 testing images.
test_iter: 1
# Carry out testing every 500 training iterations.
test_interval: 10
# The base learning rate, momentum and the weight decay of the network.
base_lr: 0.01
momentum: 0
weight_decay: 0
# The learning rate policy
lr_policy: "step"
stepsize: 5000
gamma: 0.5
# Display every 100 iterations
display: 1
# The maximum number of iterations
max_iter: 20000
# snapshot intermediate results
snapshot: 500
snapshot_prefix: "models_masklap/GT%02d/cifar10_quick_masklap"
snapshot_txt: false
# solver mode: CPU or GPU
solver_mode: CPU