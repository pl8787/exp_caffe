cd exp16_gpu_3
call train_quick_co.bat 1>train_log16_gpu_3.txt 2>&1
cd ..

cd exp32_gpu_3
call train_quick_co.bat 1>train_log32_gpu_3.txt 2>&1
cd ..

cd exp16_gpu_2
call train_quick_co.bat 1>train_log16_gpu_2.txt 2>&1
cd ..

cd exp32_gpu_2
call train_quick_co.bat 1>train_log32_gpu_2.txt 2>&1
cd ..