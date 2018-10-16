windows平台绘制FDDB的ROC曲线

1.需要安装gnuplot (我装的是gp530-20181015-win32-mingw.exe)

2.运行需要opencv_world342d.dll（你没看错，这只是个debug编译的，不过计算量不大）

3.需要安装perl（你需要在cmd里面能运行perl，我装的哪个版本已经忘了，你可以试试ActivePerl）

4.你自己的检测结果放在yourresults/detect.txt

5.运行“点此运行.bat”会生成ContROC.png, DiscROC.png