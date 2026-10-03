/* 符号探针 —— 只依赖 ZQ_CNN_Net.h，用来采集链接器眼中的未定义符号。
 *
 * 它**不是**门禁，也不参与任何判据：唯一作用是让
 * tools/gen_net_fwd_tripwires.py 有个稳定的编译单元可链接。
 *
 * 为什么要单独一个文件：门禁本身已经 include 了生成出来的绊线头，
 * 链接是通的，采集不到任何未定义符号 —— 鸡生蛋。符号集其实是
 * **ZQ_CNN_Net.h 的属性**，与门禁无关，所以探针只依赖那个头。
 *
 * 为什么必须引用 LoadFrom（两版都栽在这条上）
 * -------------------------------------------
 * 1. 只引用 `ZQ_CNN_Net::Forward` **不够**：层是**虚**函数，调用走虚表，
 *    不需要各层 Forward 的定义；层实例也没被构造过，虚表不会发射。
 *    结果探针链接**直接成功**，一个符号都采不到。
 * 2. 真正发射全部虚表的是 `_load_param_file`：那里是一条
 *    `if (name=="Convolution") new ZQ_CNN_Layer_Convolution(); else if ...`
 *    的长链，**发射这个函数就等于构造全部 36 种层**，于是 36 个虚表、
 *    全部 virtual Forward、以及它们引用的每一个 ZQ_CNN_Forward_SSEUtils
 *    辅助函数都成了未定义符号。`LoadFrom` 会调用它。
 *
 * 探针只**引用**不执行，所以这里的路径是假的。
 */
#include "ZQCNN/ZQ_CNN_Net.h"

int zq_net_symprobe(ZQ::ZQ_CNN_Net* n)
{
    return n->LoadFrom("/nonexistent.zqparams", "/nonexistent.nchwbin") ? 1 : 0;
}

int main() { return 0; }
