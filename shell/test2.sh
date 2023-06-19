
nsyns=(200000000 400000000 600000000 800000000)
nparts=(8 1 2 4) #sim time
cd ..
cd build/bench
#输出文字信息
chmod +x sim_vogel
echo "start sim vogel"
for npart in "${nparts[@]}"
do
    for nsyn in "${nsyns[@]}"
    do
        ./sim_vogel --nsyn=$nsyn --npart=$npart >> ./mgbrain_vogel_$npart.txt
        echo done:sim vogel,$npart,$nsyn
    done
done