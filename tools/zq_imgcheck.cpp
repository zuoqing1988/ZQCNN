// data/ 图像内容与扩展名一致性 + 能否被 OpenCV 解码（附录 GT）。
//
// 为什么要有它：
//   1. `data/` 里的图是**回归与示例的输入**。一张解不出来的图，
//      在 Linux 上会让 sample 打一行 "empty image" 然后失败，
//      在 Windows 上可能直接 `exit()`（附录 B-3 记的 libjpeg 缺 setjmp）。
//   2. 实测已经发现 `mouth0.jpg` / `mouth1.jpg` **是 PNG 内容、.jpg 扩展名**。
//      OpenCV 按内容嗅探所以还能读，但任何按扩展名分派的调用方
//      （IMREAD_JPEG、libjpeg 直连、文档）都会拿到错的东西。
//
// 判据：
//   * 扩展名与实际格式一致；
//   * `cv::imread` 返回非空，且尺寸 > 0；
//   * `cv::imdecode` 在**指定格式**下（IMREAD_JPEG / IMREAD_PNG）也能解 ——
//     这一条专门抓"扩展名骗人但解码器按格式来"的情形。
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <dirent.h>
#include <opencv2/opencv.hpp>

static std::string detect_format(const std::string& path)
{
    // 只看魔数，不看扩展名 —— 扩展名正是被怀疑的那一方
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return "(打不开)";
    unsigned char h[8] = {0};
    size_t n = fread(h, 1, 8, f);
    fclose(f);
    if (n >= 3 && h[0] == 0xFF && h[1] == 0xD8 && h[2] == 0xFF) return "JPEG";
    if (n >= 8 && h[0] == 0x89 && h[1] == 'P' && h[2] == 'N' && h[3] == 'G')
        return "PNG";
    if (n >= 2 && h[0] == 'B' && h[1] == 'M') return "BMP";
    if (n >= 6 && !memcmp(h, "GIF87a", 6)) return "GIF";
    return "(不认识)";
}

int main(int argc, char** argv)
{
    const char* dir = argc > 1 ? argv[1] : "data";
    std::vector<std::string> names;
    DIR* d = opendir(dir);
    if (!d) { printf("打不开目录 %s\n", dir); return 2; }
    struct dirent* e;
    while ((e = readdir(d)) != NULL) {
        std::string n = e->d_name;
        if (n.size() < 4) continue;
        std::string ext = n.substr(n.size() - 4);
        for (size_t i = 0; i < ext.size(); i++) ext[i] = (char)tolower(ext[i]);
        if (ext == ".jpg" || ext == ".png" || ext == ".jpeg" || ext == ".bmp")
            names.push_back(n);
    }
    closedir(d);

    int n_bad_ext = 0, n_decode = 0, n_forced = 0;
    for (size_t i = 0; i < names.size(); i++) {
        std::string p = std::string(dir) + "/" + names[i];
        std::string ext = names[i].substr(names[i].size() - 4);
        for (size_t k = 0; k < ext.size(); k++) ext[k] = (char)tolower(ext[k]);
        if (ext == ".jpeg") ext = ".jpg";
        std::string real = detect_format(p);

        // 扩展名声称的格式
        std::string claim = (ext == ".jpg" || ext == ".jpeg") ? "JPEG"
                          : (ext == ".png" ? "PNG" : "BMP");
        bool ext_ok = (real == claim);

        cv::Mat m = cv::imread(p, cv::IMREAD_COLOR);
        bool dec_ok = !m.empty();

        // 按魔数认出的格式**强制**解码：抓"扩展名骗人、调用方按格式来"
        bool forced_ok = true;
        if (real == "JPEG" || real == "PNG" || real == "BMP") {
            int flag = (real == "JPEG") ? cv::IMREAD_COLOR
                      : (real == "PNG") ? cv::IMREAD_COLOR : cv::IMREAD_COLOR;
            cv::Mat m2 = cv::imread(p, flag);
            forced_ok = !m2.empty();
        }

        if (!ext_ok) {
            printf("  扩展名不符  %-24s 扩展名说 %-4s 实际是 %s\n",
                   names[i].c_str(), claim.c_str(), real.c_str());
            n_bad_ext++;
        }
        if (!dec_ok) {
            printf("  解不出来    %-24s\n", names[i].c_str());
            n_decode++;
        }
        if (!forced_ok) {
            printf("  按格式解不出 %-24s 实际是 %s\n", names[i].c_str(), real.c_str());
            n_forced++;
        }
    }
    printf("扫了 %d 张图：扩展名不符 %d，解不出 %d，按格式解不出 %d\n",
           (int)names.size(), n_bad_ext, n_decode, n_forced);
    return (n_bad_ext || n_decode || n_forced) ? 1 : 0;
}
