#!/usr/bin/env sh
set path="D:\CppDevTools\ZeroMQ 4.0.4\bin;D:\Python27\;C:\Program Files\Microsoft HPC Pack 2012\Bin\;C:\Program Files\Microsoft MPI\Bin\;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.5\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.5\libnvvp;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0\bin\;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0\libnvvp\;C:\windows\system32;C:\windows;C:\windows\System32\Wbem;C:\windows\System32\WindowsPowerShell\v1.0\;C:\Program Files\Microsoft SQL Server\110\DTS\Binn\;C:\Program Files (x86)\Microsoft SQL Server\110\Tools\Binn\;C:\Program Files\Microsoft SQL Server\110\Tools\Binn\;C:\Program Files (x86)\Microsoft SQL Server\110\Tools\Binn\ManagementStudio\;C:\Program Files (x86)\Microsoft Visual Studio 10.0\Common7\IDE\PrivateAssemblies\;C:\Program Files (x86)\Microsoft SQL Server\110\DTS\Binn\;C:\Program Files (x86)\Windows Kits\8.1\Windows Performance Toolkit\;C:\Program Files\Microsoft\Web Platform Installer\;C:\Program Files (x86)\Microsoft ASP.NET\ASP.NET Web Pages\v1.0\;D:\require\alldll\;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.5\;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0\;D:\require\intel64\mkl\;D:\require\Intel\mkl\redist\intel64\compiler\;D:\require\intel64\compiler\;D:\tools\MATLAB.R2012a.X64.Portable\bin\win64\;C:\CTEX\UserData\miktex\bin;C:\CTEX\MiKTeX\miktex\bin;C:\CTEX\CTeX\ctex\bin;C:\CTEX\CTeX\cct\bin;C:\CTEX\CTeX\ty\bin;C:\CTEX\Ghostscript\gs9.05\bin;C:\CTEX\GSview\gsview;C:\CTEX\WinEdt;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.5\bin"

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/train_net.exe cifar10_quick_solver.prototxt

#reduce learning rate by fctor of 10 after 8 epochs
GLOG_logtostderr=1 $TOOLS/train_net.exe cifar10_quick_solver_lr1.prototxt cifar10_quick_iter_4000.solverstate
