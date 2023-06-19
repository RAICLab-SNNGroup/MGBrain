cd ..
cd build/bench
chmod +x opart_brunel
models=('bsim' 'metis' 'model')
nparts=(2 4 8)
nsyns=(10000000 20000000 30000000 40000000 50000000 60000000 70000000 80000000 90000000)
for model in "${models[@]}" 
do 
    for part in "${nparts[@]}"
    do
        for nsyn in "${nsyns[@]}"
        do
            ./opart_brunel --model=$model --npart=$part --nsyn=$nsyn
            echo done:$model,$part,$nsyn
        done
    done
done


