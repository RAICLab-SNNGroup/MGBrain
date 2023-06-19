cd ..
cd build/bench
chmod +x single_brunel
# nsyns=(10000000 20000000 30000000 40000000 50000000 60000000 70000000 80000000 90000000 100000000)
nsyns=(20000000 40000000 60000000 80000000 100000000 120000000 140000000 160000000 180000000 200000000)
rates=(0.002 0.01 0.05 0.1 0.2)
# nsyns=(120000000 140000000 160000000 180000000)
# rates=(0.002)
# 输出文字信息
echo "start sim single"
for nsyn in "${nsyns[@]}"
do
    for rate in "${rates[@]}"
    do
        nvprof ./single_brunel --nsyn=$nsyn --rate=$rate > ../../benchdata/single/mgbrain_single_brunel_${nsyn}_${rate}_nvprof.txt 2>&1
        echo done:$nsyn,$rate
    done
done
#输出nvvp文件信息
# for nsyn in "${nsyns[@]}"
# do
#     nvprof -o /home/cuimy/zfz/MGBrain-new/benchdata/single/mgbrain_brunel_${nsyn}_nvprof.nvvp /home/cuimy/zfz/MGBrain-new/build/bench/single_brunel --nsyn=$nsyn
#     echo done:$nsyn
# done


