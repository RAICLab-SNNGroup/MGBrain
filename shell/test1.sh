cd ..
cd build/bench
chmod +x single_vogel
nsyns=(10000000 20000000 30000000 40000000 50000000 60000000 70000000 80000000 90000000 100000000)


# 输出文字信息
for nsyn in "${nsyns[@]}"
do
    nvprof /home/cuimy/zfz/MGBrain-new/build/bench/single_vogel --nsyn=$nsyn 2>&1 | tee /home/cuimy/zfz/MGBrain-new/benchdata/single/mgbrain_vogel_${nsyn}_nvprof.txt
    echo done:$nsyn
done
#输出nvvp文件信息
# for nsyn in "${nsyns[@]}"
# do
#     nvprof -o /home/cuimy/zfz/MGBrain-new/benchdata/single/mgbrain_vogel_${nsyn}_nvprof.nvvp /home/cuimy/zfz/MGBrain-new/build/bench/single_vogel --nsyn=$nsyn
#     echo done:$nsyn
# done