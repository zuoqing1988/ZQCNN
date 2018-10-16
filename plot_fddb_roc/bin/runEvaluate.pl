#!/usr/bin/perl -w

use strict;

#### VARIABLES TO EDIT ####
# where gnuplot is
#my $GNUPLOT = "/sw/bin/gnuplot"; 
#my $GNUPLOT = "\"C:/Program Files (x86)/gnuplot/bin/gnuplot\""; 
# where the binary is
#my $evaluateBin = "x64/debug/evaluate"; 
# where the images are
#my $imDir = "G:/mtcnn/data/fddb/images/"; 
# where the folds are
#my $fddbDir = "G:/mtcnn/data/fddb/imglists/"; 
# where the detections are
my $detDir = "./"; 
###########################

my $detFormat = 0; # 0: rectangle, 1: ellipse 2: pixels


sub makeGNUplotFile
{
  my $rocFile = shift;
  my $gnuplotFile = shift;
  my $title = shift;
  my $pngFile = shift;
  open(GF, ">$gnuplotFile") or die "Can not open $gnuplotFile for writing\n";
  #print GF "$GNUPLOT\n";
  print GF "set term png\n";
  print GF "set size .75,1\n";
  print GF "set output \"$pngFile\"\n";
  #print GF "set xtics 100\n";
  #print GF "set logscale x\n";
  print GF "set ytics .1\n";
  print GF "set grid\n";
  #print GF "set size ratio -1\n";
  print GF "set ylabel \"True positive rate\"\n";
  print GF "set xlabel \"False positives\"\n";
  #print GF "set xr [0:500]\n";
  print GF "set yr [0:1]\n";
  print GF "set key right bottom\n";
  print GF "plot \"$rocFile\" using 2:1 title \"$title\" with lines lw 2 \n";
  close(GF);
}
my $gpFile = "./ContROC.p";
my $gpFile1 = "./DiscROC.p";
my $title = "title";
# plot the two ROC curves using GNUplot
makeGNUplotFile("./tempContROC.txt", $gpFile, $title, $detDir."ContROC.png");
makeGNUplotFile("./tempDiscROC.txt", $gpFile1, $title, $detDir."DiscROC.png");
