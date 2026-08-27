// ============================================================
// blue_candidates.cpp  (簡化寬鬆版 / FIXED contour->template order)
//
// 根本 bug 修正：
//   舊：kContourToTemplate = {0, 2, 3, 4, 5, 7, 6, 1}   <-- 錯
//   新：kContourToTemplate = {0, 2, 3, 1, 6, 4, 5, 7}   <-- 對
//
// 理由（已用數值驗證）：
//   C 的實體形狀 = 左邊直邊 + 右側中央挖凹槽的字母 C。
//   外框角點: P0(左上) P2(右上) P5(右下) P7(左下)
//   凹槽角點: P1, P6 (內角)  / P3, P4 (凹槽出入口)
//   真正沿輪廓走一圈: P0 -> P2 -> P3 -> P1 -> P6 -> P4 -> P5 -> P7
//
//   用舊映射算 Contour2 的 H RMS = 19.7px
//   用新映射算 Contour2 的 H RMS = 0.70px  <- 幾乎完美
//
// 本版另外做的簡化（先求「抓到」，再收緊「精準」）：
//   1. 拿掉 80-step epsilon 搜尋，只用單一 epsilon（0.01 * perimeter）
//      正確映射下不需要暴力多組 epsilon 搜尋。
//   2. Debug 輸出精簡，只印關鍵資訊。
//   3. Threshold 暫時放寬（kRelaxedMaxRms），方便你先確認能抓到，
//      之後再改回 config::kMaxHomographyRms / kMaxPnPRms。
// ============================================================

#include "core/blue_candidates.hpp"
#include "core/config.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace tel::core {
namespace {

constexpr double kEpsilon = 1e-9;
constexpr bool kDebugBlue = true;
constexpr int kN = 8;

// ------------------------------------------------------------
// 暫時放寬的驗證門檻（先確認 pipeline 抓得到，之後改回 config 裡的值）
// ------------------------------------------------------------
constexpr double kRelaxedMaxHomographyRms = 8.0;
constexpr double kRelaxedMaxPnPRms        = 8.0;

// ------------------------------------------------------------
// FIXED: 正確的 contour walk order -> kCTemplate index
// ------------------------------------------------------------
constexpr std::array<int, kN> kContourToTemplate = {
    0, 2, 3, 1, 6, 4, 5, 7
};

constexpr double kCHeightModel = 46.467 - (-25.533);
constexpr double kModelUnitToMm = 116.0 / kCHeightModel;

std::array<cv::Point2d, kN> buildCModelMm()
{
    std::array<cv::Point2d, kN> model{};
    for (int i = 0; i < kN; ++i) {
        const auto p = config::kCTemplate[kContourToTemplate[i]];
        model[i] = cv::Point2d(p.x * kModelUnitToMm, p.y * kModelUnitToMm);
    }
    return model;
}

std::array<cv::Point3f, kN> buildCModel3D()
{
    const auto model2d = buildCModelMm();
    std::array<cv::Point3f, kN> model{};
    for (int i = 0; i < kN; ++i) {
        model[i] = cv::Point3f(
            static_cast<float>(model2d[i].x),
            static_cast<float>(model2d[i].y),
            0.0f);
    }
    return model;
}

cv::Point3f buildGreenModelPoint()
{
    return cv::Point3f(
        static_cast<float>(config::kGreenCenterFromOrigin.x * kModelUnitToMm),
        static_cast<float>(config::kGreenCenterFromOrigin.y * kModelUnitToMm),
        0.0f);
}

std::vector<cv::Point2f> toVector(const std::array<cv::Point2f, kN>& pts)
{
    return {pts.begin(), pts.end()};
}

std::array<cv::Point2f, kN> reorderPolygon(
    const std::vector<cv::Point>& polygon, int offset, bool reverse)
{
    std::array<cv::Point2f, kN> result{};
    for (int i = 0; i < kN; ++i) {
        int index = reverse ? (offset - i + kN * 2) % kN
                             : (offset + i) % kN;
        result[i] = cv::Point2f(
            static_cast<float>(polygon[index].x),
            static_cast<float>(polygon[index].y));
    }
    return result;
}

struct HomographyResult {
    bool valid = false;
    cv::Mat H;
    double rms = std::numeric_limits<double>::max();
    cv::Point2f expectedGreen{0.f, 0.f};
};

bool projectHomographyPoint(const cv::Mat& H, const cv::Point2d& src, cv::Point2f& dst)
{
    if (H.empty() || H.rows != 3 || H.cols != 3) return false;
    const double x = src.x, y = src.y;
    const double w = H.at<double>(2,0)*x + H.at<double>(2,1)*y + H.at<double>(2,2);
    if (std::abs(w) < kEpsilon) return false;
    const double u = (H.at<double>(0,0)*x + H.at<double>(0,1)*y + H.at<double>(0,2)) / w;
    const double v = (H.at<double>(1,0)*x + H.at<double>(1,1)*y + H.at<double>(1,2)) / w;
    if (!std::isfinite(u) || !std::isfinite(v)) return false;
    dst = cv::Point2f(static_cast<float>(u), static_cast<float>(v));
    return true;
}

HomographyResult calculateHomography(const std::array<cv::Point2f, kN>& imagePoints)
{
    HomographyResult result;
    const auto model = buildCModelMm();
    std::vector<cv::Point2f> modelPoints;
    modelPoints.reserve(kN);
    for (const auto& p : model)
        modelPoints.emplace_back(static_cast<float>(p.x), static_cast<float>(p.y));

    cv::Mat H = cv::findHomography(modelPoints, toVector(imagePoints), 0);
    if (H.empty()) return result;
    H.convertTo(result.H, CV_64F);

    std::vector<cv::Point2f> projected;
    cv::perspectiveTransform(modelPoints, projected, result.H);
    if (projected.size() != kN) return result;

    double sum = 0.0;
    for (int i = 0; i < kN; ++i) {
        const double dx = projected[i].x - imagePoints[i].x;
        const double dy = projected[i].y - imagePoints[i].y;
        sum += dx*dx + dy*dy;
    }
    result.rms = std::sqrt(sum / kN);
    if (!std::isfinite(result.rms)) return result;

    const auto green = config::kGreenCenterFromOrigin;
    const cv::Point2d greenMm{green.x * kModelUnitToMm, green.y * kModelUnitToMm};
    if (!projectHomographyPoint(result.H, greenMm, result.expectedGreen)) return result;

    result.valid = result.rms <= kRelaxedMaxHomographyRms;
    return result;
}

struct PnPResult {
    bool valid = false;
    cv::Vec3d rvec{0,0,0};
    cv::Vec3d tvec{0,0,0};
    double rms = std::numeric_limits<double>::max();
    cv::Point2f expectedGreen{0.f, 0.f};
};

PnPResult calculatePnP(const std::array<cv::Point2f, kN>& imagePoints)
{
    PnPResult result;
    const auto modelArray = buildCModel3D();
    const std::vector<cv::Point3f> objectPoints(modelArray.begin(), modelArray.end());
    const auto imageVector = toVector(imagePoints);
    const cv::Matx33d cameraMatrix = config::kFallbackCameraMatrix;

    cv::Mat distCoeffs(static_cast<int>(config::kFallbackDist.size()), 1, CV_64F);
    for (int i = 0; i < static_cast<int>(config::kFallbackDist.size()); ++i)
        distCoeffs.at<double>(i,0) = config::kFallbackDist[i];

    try {
        if (!cv::solvePnP(objectPoints, imageVector, cameraMatrix, distCoeffs,
                           result.rvec, result.tvec, false, cv::SOLVEPNP_SQPNP))
            return result;
    } catch (const cv::Exception&) { return result; }

    for (int i = 0; i < 3; ++i)
        if (!std::isfinite(result.rvec[i]) || !std::isfinite(result.tvec[i])) return result;
    if (result.tvec[2] <= 0.0) return result;

    std::vector<cv::Point2f> projected;
    try {
        cv::projectPoints(objectPoints, result.rvec, result.tvec, cameraMatrix, distCoeffs, projected);
    } catch (const cv::Exception&) { return result; }
    if (projected.size() != kN) return result;

    double sum = 0.0;
    for (int i = 0; i < kN; ++i) {
        const double dx = projected[i].x - imagePoints[i].x;
        const double dy = projected[i].y - imagePoints[i].y;
        sum += dx*dx + dy*dy;
    }
    result.rms = std::sqrt(sum / kN);
    if (!std::isfinite(result.rms)) return result;

    const cv::Point3f green = buildGreenModelPoint();
    std::vector<cv::Point3f> greenObject{green};
    std::vector<cv::Point2f> greenProjected;
    try {
        cv::projectPoints(greenObject, result.rvec, result.tvec, cameraMatrix, distCoeffs, greenProjected);
    } catch (const cv::Exception&) { return result; }
    if (greenProjected.size() != 1) return result;

    result.expectedGreen = greenProjected[0];
    result.valid = result.rms <= kRelaxedMaxPnPRms;
    return result;
}

struct BestGeometry {
    bool found = false;
    std::array<cv::Point2f, kN> points{};
    HomographyResult homography;
    PnPResult pnp;
    double score = std::numeric_limits<double>::max();
    int reverse = 0, offset = 0;
};

BestGeometry findBestGeometry(const std::vector<cv::Point>& polygon)
{
    BestGeometry best;
    for (int reverse = 0; reverse <= 1; ++reverse) {
        for (int offset = 0; offset < kN; ++offset) {
            const auto points = reorderPolygon(polygon, offset, reverse != 0);
            const auto h = calculateHomography(points);
            const auto p = calculatePnP(points);

            if (kDebugBlue) {
                std::cout << "    [GEOM] rev=" << reverse << " off=" << offset
                          << " H=" << (h.valid ? "OK" : "FAIL") << " RMS=" << h.rms
                          << " PnP=" << (p.valid ? "OK" : "FAIL") << " RMS=" << p.rms << '\n';
            }

            double score = h.rms + p.rms;
            if (!std::isfinite(score)) continue;

            if (!best.found || score < best.score) {
                best = {true, points, h, p, score, reverse, offset};
            }
        }
    }
    return best;
}

} // namespace

std::vector<BlueCandidate> find_blue_candidates(const cv::Mat& blue_mask)
{
    std::vector<BlueCandidate> candidates;
    if (blue_mask.empty()) return candidates;

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(blue_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (kDebugBlue) {
        std::cout << "[BLUE] contours=" << contours.size()
                  << " mm/unit=" << kModelUnitToMm << '\n';
    }

    for (std::size_t ci = 0; ci < contours.size(); ++ci) {
        const auto& contour = contours[ci];
        const double area = cv::contourArea(contour);
        if (area < config::kMinBlueArea) continue;
        if (contour.size() < config::kMinContourPoints) continue;

        const double perimeter = cv::arcLength(contour, true);
        if (perimeter <= 0.0) continue;

        std::vector<cv::Point> polygon;
        cv::approxPolyDP(contour, polygon, config::kApproxEpsilonRatio * perimeter, true);

        if (polygon.size() != kN) {
            if (kDebugBlue)
                std::cout << "[BLUE][Contour " << ci << "] approx=" << polygon.size()
                          << " != 8, skip\n";
            continue;
        }

        if (kDebugBlue) std::cout << "\n[BLUE][Contour " << ci << "] area=" << area << '\n';

        const auto geometry = findBestGeometry(polygon);
        if (!geometry.found) continue;

        const double greenDelta = cv::norm(
            geometry.homography.expectedGreen - geometry.pnp.expectedGreen);

        BlueCandidate candidate;
        candidate.contour = contour;
        candidate.area = area;
        candidate.points = geometry.points;
        candidate.homography = geometry.homography.H;
        candidate.expected_green_h = geometry.homography.expectedGreen;
        candidate.homography_reprojection_rms = geometry.homography.rms;
        candidate.homography_valid = geometry.homography.valid;
        candidate.rvec = geometry.pnp.rvec;
        candidate.tvec = geometry.pnp.tvec;
        candidate.expected_green_pnp = geometry.pnp.expectedGreen;
        candidate.pnp_reprojection_rms = geometry.pnp.rms;
        candidate.pnp_valid = geometry.pnp.valid;
        candidate.green_prediction_error = greenDelta;
        candidate.valid = candidate.homography_valid && candidate.pnp_valid &&
            std::isfinite(greenDelta) && greenDelta <= config::kMaxGreenPredictionError;

        if (kDebugBlue) {
            std::cout << "  BEST rev=" << geometry.reverse << " off=" << geometry.offset
                      << " H_RMS=" << geometry.homography.rms
                      << " PnP_RMS=" << geometry.pnp.rms
                      << " valid=" << candidate.valid << '\n';
        }

        candidates.push_back(std::move(candidate));
    }

    return candidates;
}

} // namespace tel::core