nsyns=(250000000 500000000 750000000 1000000000)
nparts=(1 2 4 8)
rates=(0.01 0.05 0.1 0.2)
# nsyns=(1000000000)
# nparts=(8)
# rates=(0.01)
cd ..
cd build/bench
#输出文字信息
chmod +x sim_brunel
echo "start sim brunel"
for npart in "${nparts[@]}"
do
    for nsyn in "${nsyns[@]}"
    do
        for rate in "${rates[@]}"
        do
            ./sim_brunel --net=brunel --nsyn=$nsyn --npart=$npart --rate=$rate >> ./mgbrain_brunel_$npart.txt
            echo done:sim brunel,$npart,$nsyn,$rate
        done 
    done
done

