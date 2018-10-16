set term png
set size .75,1
set output "./DiscROC.png"
set ytics .1
set grid
set ylabel "True positive rate"
set xlabel "False positives"
set yr [0:1]
set key right bottom
plot "./tempDiscROC.txt" using 2:1 title "title" with lines lw 2 
