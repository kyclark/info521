all: mens1 synth womens cv

clean:
	rm -f info521*.{aux,log,gz} *.png

mens1:
	./fitpoly.py mens100.csv -t "Men's 100 1st order" -x 'Year' -o mens-1.png 

mens6:
	./fitpoly.py mens100.csv -t "Men's 100 6th order (not scaled)" -x 'Year' -o mens-6.png -m 6

mens6scale:
	./fitpoly.py mens100.csv -t "Men's 100 6th order (scaled)" -x 'Year' -o mens-6-scaled.png -m 6 -s

synth:
	./fitpoly.py synthdata2018.csv -t "Synthetic" -m 3 -o synth.png

womens:
	./fitpoly.py womens100.csv -t "Women's 100" -x 'Year' -o womens.png

cv:
	./cv.py synthdata2018.csv -k 5 -m 7 -o synth_cv.png -n 5
