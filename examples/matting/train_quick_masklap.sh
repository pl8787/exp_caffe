#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/train_net.exe cifar10_quick_solver_masklap.prototxt

