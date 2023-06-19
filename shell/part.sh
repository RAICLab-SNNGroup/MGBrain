cd ..
cd build/bench
chmod +x part_brunel
models=('bsim' 'metis' 'model')
nparts=(2 4 8)
nsyns=(200000000 400000000 600000000)
for model in "${models[@]}" 
do 
    for part in "${nparts[@]}"
    do
        for nsyn in "${nsyns[@]}"
        do
            ./part_brunel --model=$model --npart=$part --nsyn=$nsyn --rate=0.01 >> ./mgbrain_part_brunel_$nsyn.txt
            echo done:part brunel,$model,$part
        done
    done
done