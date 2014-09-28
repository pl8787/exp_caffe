REM cd exp16_gpu_base
REM call train_quick.bat 1>train_log16_gpu_base.txt 2>&1
REM cd ..

cd exp32_gpu_base
call train_quick.bat 1>train_log32_gpu_base.txt 2>&1
cd ..

REM cd exp16_gpu_3
REM call train_quick_co.bat 1>train_log16_gpu_3.txt 2>&1
REM cd ..

REM cd exp32_gpu_3
REM call train_quick_co.bat 1>train_log32_gpu_3.txt 2>&1
REM cd ..

REM cd exp16_gpu_2
REM call train_quick_co.bat 1>train_log16_gpu_2.txt 2>&1
REM cd ..

REM cd exp32_gpu_2
REM call train_quick_co.bat 1>train_log32_gpu_2.txt 2>&1
REM cd ..