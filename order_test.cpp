// order_test.cpp — 診斷工具
// 用真實 log 的 C 輪廓，暴力測試所有 8! 排列 + 特定對應 [0,7,5,4,6,1,3,2]
#include <algorithm>
#include <array>
#include <cstdio>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

static const std::array<cv::Point2d, 8> kTemplate = {{
    {-23.500, -25.533},
    {-7.481, -10.233},
    {24.500, -25.533},
    {24.000, -11.521},
    {23.814, 30.967},
    {23.500, 46.467},
    {-7.474, 28.967},
    {-23.500, 46.467},
}};

static const cv::Matx33d kCam{1112.0566597, 0.0, 954.8446657, 0.0, 1109.8558990,
                              489.2472785,  0.0, 0.0,         1.0};

static double homographyRms(const std::vector<cv::Point2f> &model,
                            const std::vector<cv::Point2f> &img) {
  cv::Mat H = cv::findHomography(model, img, 0);
  if (H.empty())
    return 1e9;
  std::vector<cv::Point2f> proj;
  cv::perspectiveTransform(model, proj, H);
  double sum = 0.0;
  for (int i = 0; i < 8; ++i) {
    const double dx = proj[i].x - img[i].x;
    const double dy = proj[i].y - img[i].y;
    sum += dx * dx + dy * dy;
  }
  return std::sqrt(sum / 8.0);
}

static double pnpRms(const std::vector<cv::Point2f> &img) {
  std::vector<cv::Point3f> obj;
  for (const auto &p : kTemplate)
    obj.emplace_back(static_cast<float>(p.x), static_cast<float>(p.y), 0.f);
  cv::Mat dist(5, 1, CV_64F);
  const double d[5] = {0.0399809499, -0.1098228779, 0.0008024654, -0.0012401027,
                       0.0505635206};
  for (int i = 0; i < 5; ++i)
    dist.at<double>(i, 0) = d[i];
  cv::Mat rvec, tvec;
  try {
    if (!cv::solvePnP(obj, img, kCam, dist, rvec, tvec, false,
                      cv::SOLVEPNP_SQPNP))
      return 1e9;
  } catch (...) {
    return 1e9;
  }
  std::vector<cv::Point2f> proj;
  cv::projectPoints(obj, rvec, tvec, kCam, dist, proj);
  double sum = 0.0;
  for (int i = 0; i < 8; ++i) {
    const double dx = proj[i].x - img[i].x;
    const double dy = proj[i].y - img[i].y;
    sum += dx * dx + dy * dy;
  }
  return std::sqrt(sum / 8.0);
}

static bool isReverseOffset(const std::array<int, 8> &order) {
  for (int reverse = 0; reverse <= 1; ++reverse) {
    for (int offset = 0; offset < 8; ++offset) {
      bool ok = true;
      for (int i = 0; i < 8; ++i) {
        int idx = !reverse ? (offset + i) % 8 : (offset - i + 16) % 8;
        if (order[i] != idx) {
          ok = false;
          break;
        }
      }
      if (ok)
        return true;
    }
  }
  return false;
}

static void test(const char *name, const std::vector<cv::Point2f> &img) {
  const std::vector<cv::Point2f> model(kTemplate.begin(), kTemplate.end());

  std::array<int, 8> perm = {0, 1, 2, 3, 4, 5, 6, 7};
  double bestH = 1e9, bestP = 1e9;
  std::array<int, 8> bestHp = perm, bestPp = perm;

  do {
    std::vector<cv::Point2f> ordered(8);
    for (int i = 0; i < 8; ++i)
      ordered[i] = img[perm[i]];
    const double h = homographyRms(model, ordered);
    const double p = pnpRms(ordered);
    if (h < bestH) {
      bestH = h;
      bestHp = perm;
    }
    if (p < bestP) {
      bestP = p;
      bestPp = perm;
    }
  } while (std::next_permutation(perm.begin(), perm.end()));

  // 特定對應：img[A,B,C,D,E,F,G,H] -> template[P0,P7,P5,P4,P6,P1,P3,P2]
  const std::array<int, 8> specific = {0, 7, 5, 4, 6, 1, 3, 2};
  std::vector<cv::Point2f> s(8);
  for (int i = 0; i < 8; ++i)
    s[i] = img[specific[i]];

  auto printPerm = [](const std::array<int, 8> &a) {
    for (int i = 0; i < 8; ++i)
      std::printf("%d ", a[i]);
  };

  std::printf("[%s]\n", name);
  std::printf("  img order -> template idx:  A B C D E F G H\n");
  std::printf("  specific [0,7,5,4,6,1,3,2]: H RMS=%.3f px | PnP RMS=%.3f px\n",
              homographyRms(model, s), pnpRms(s));
  std::printf("  best over 8!  H : ");
  printPerm(bestHp);
  std::printf(" | RMS=%.3f px | in16=%s\n", bestH,
              isReverseOffset(bestHp) ? "YES" : "NO");
  std::printf("  best over 8! PnP : ");
  printPerm(bestPp);
  std::printf(" | RMS=%.3f px | in16=%s\n\n", bestP,
              isReverseOffset(bestPp) ? "YES" : "NO");
}

int main() {
  // frame 4 的兩個通過點數檢查的 C
  const std::vector<cv::Point2f> c6 = {{676, 240}, {676, 313}, {724, 311},
                                       {723, 296}, {692, 295}, {694, 255},
                                       {723, 253}, {721, 240}};
  const std::vector<cv::Point2f> c7 = {{659, 240}, {604, 246}, {606, 259},
                                       {622, 259}, {626, 315}, {643, 315},
                                       {643, 259}, {661, 256}};
  test("Frame4 Contour 6", c6);
  test("Frame4 Contour 7", c7);
  return 0;
}
