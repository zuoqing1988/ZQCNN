set prefix=model-emore-fine
set epoch=0
python json2prototxt.py --mx-json %prefix%-symbol.json --cf-prototxt model_tmp.prototxt

setlocal enabledelayedexpansion
@echo off
echo replace _mulscalar0 with data
set f1=model_tmp.prototxt
set f2="model.prototxt"
if exist %f2% del %f2% 
for /f "tokens=* delims=ге" %%l in (%f1%) do ( 
 set line=%%l 
 set line=!line:_mulscalar0=data! 
 echo !line!>>%f2% 
 )
echo replace done! 

@echo on

python mxnet2caffe.py --mx-model %prefix% --mx-epoch %epoch% --cf-prototxt model.prototxt --cf-model model.caffemodel
pause