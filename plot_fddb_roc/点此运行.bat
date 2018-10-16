set baseDir=G:/mtcnn/data/fddb/images/
set imList=groundtruth/imList.txt
set yours=yourresults/detect.txt
set truth=groundtruth/ellipseList.txt
set myGNUPLOT="C:/Program Files (x86)/gnuplot/bin/gnuplot"

bin\EvaluationFDDB %baseDir% %imList% %yours% %truth%

perl bin\runEvaluate.pl
%myGNUPLOT% ContROC.p
%myGNUPLOT% DiscROC.p
pause