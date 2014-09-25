#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/train_net.exe cifar10_quick_solver_co.prototxt

#reduce learning rate by fctor of 10 after 8 epochs
GLOG_logtostderr=1 $TOOLS/train_net.exe cifar10_quick_solver_lr1_co.prototxt cifar10_quick_co_iter_4000.solverstate
