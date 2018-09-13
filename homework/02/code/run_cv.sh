for i in $(seq 1 5); do
    #./cv.py synthdata2018.csv -k 5 -m 7 -o cv_synth_${i}.png -r -n
    ./cv.py synthdata2018.csv -k 19 -m 7 -o loocv_synth_${i}.png -r -n
done
