// ============================================================
// blue_candidates.cpp
//
// 職責：
//   從 blue binary mask 找出所有 C。
//   每個 C 必須：
//      1. 有效 contour
//      2. approxPolyDP = 8 points
//      3. Homography 幾何驗證
//      4. SQPnP 幾何驗證
//      5. 計算理論 Green center
//
// 注意：
//   本模組不負責尋找 / 匹配實際 Green。
//   它只回答：
//
//       「如果這個 C 是真的，那 Green 理論上應該在哪裡？」
//
//   Pipeline 再拿 expected_green_h / expected_green_pnp
//   與實際 green_center 比較。
// ============================================================

#include "core/blue_candidates.hpp"
#include "core/config.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <iomanip>
#include <sstream>

namespace tel::core {
namespace {

// ============================================================
// Constants
// ============================================================

constexpr double kEpsilon = 1e-9;
constexpr bool kDebugBlue = true;

// ============================================================
// C model -> 3D model
//
// C 固定在 Z = 0 平面。
// ============================================================

std::array<cv::Point3f, 8> buildCModel3D()
{
    std::array<cv::Point3f, 8> model{};

    for (std::size_t i = 0; i < 8; ++i) {

        model[i] = cv::Point3f(
            static_cast<float>(
                config::kCTemplate[i].x
            ),
            static_cast<float>(
                config::kCTemplate[i].y
            ),
            0.0f
        );
    }

    return model;
}


// ============================================================
// Green model point
//
// 與 C template 使用同一座標系。
// ============================================================

cv::Point3f buildGreenModelPoint()
{
    return cv::Point3f(
        static_cast<float>(
            config::kGreenCenterFromOrigin.x
        ),
        static_cast<float>(
            config::kGreenCenterFromOrigin.y
        ),
        0.0f
    );
}


// ============================================================
// Convert array<Point2f,8> -> vector<Point2f>
// ============================================================

std::vector<cv::Point2f>
toVector(
    const std::array<cv::Point2f, 8>& points
)
{
    return std::vector<cv::Point2f>(
        points.begin(),
        points.end()
    );
}


// ============================================================
// Polygon ordering
//
// approxPolyDP 給出的 8 點本身會沿 contour 排列，
// 但是：
//
//     P0 不一定對應 template P0
//
// 所以必須找 cyclic offset。
//
// 同時測試正向 / 反向，避免 OpenCV contour winding
// 與 template 不一致。
//
// 這不是「C 匹配枚舉」。
// 它只是解決同一個 C 的 polygon index ambiguity。
// ============================================================

std::array<cv::Point2f, 8>
reorderPolygon(
    const std::vector<cv::Point>& polygon,
    int offset,
    bool reverse
)
{
    std::array<cv::Point2f, 8> result{};

    constexpr int N = 8;

    for (int i = 0; i < N; ++i) {

        int index;

        if (!reverse) {
            index = (offset + i) % N;
        }
        else {
            index = (offset - i + N * 2) % N;
        }

        result[i] = cv::Point2f(
            static_cast<float>(
                polygon[index].x
            ),
            static_cast<float>(
                polygon[index].y
            )
        );
    }

    return result;
}


// ============================================================
// Calculate RMS between two 2D point arrays
// ============================================================

double calculateRms(
    const std::array<cv::Point2f, 8>& a,
    const std::array<cv::Point2f, 8>& b
)
{
    double sum = 0.0;

    for (int i = 0; i < 8; ++i) {

        const double dx =
            static_cast<double>(a[i].x) -
            static_cast<double>(b[i].x);

        const double dy =
            static_cast<double>(a[i].y) -
            static_cast<double>(b[i].y);

        sum += dx * dx + dy * dy;
    }

    return std::sqrt(
        sum / 8.0
    );
}


// ============================================================
// Homography projection
// ============================================================

bool projectHomographyPoint(
    const cv::Mat& H,
    const cv::Point2d& source,
    cv::Point2f& destination
)
{
    if (H.empty())
        return false;

    if (H.rows != 3 || H.cols != 3)
        return false;

    const double x = source.x;
    const double y = source.y;

    const double w =
        H.at<double>(2, 0) * x +
        H.at<double>(2, 1) * y +
        H.at<double>(2, 2);

    if (std::abs(w) < kEpsilon)
        return false;

    const double u =
        (
            H.at<double>(0, 0) * x +
            H.at<double>(0, 1) * y +
            H.at<double>(0, 2)
        ) / w;

    const double v =
        (
            H.at<double>(1, 0) * x +
            H.at<double>(1, 1) * y +
            H.at<double>(1, 2)
        ) / w;

    if (!std::isfinite(u) ||
        !std::isfinite(v))
        return false;

    destination = cv::Point2f(
        static_cast<float>(u),
        static_cast<float>(v)
    );

    return true;
}


// ============================================================
// Homography evaluation
// ============================================================

struct HomographyResult {

    bool valid = false;

    cv::Mat H;

    double rms = 0.0;

    cv::Point2f expectedGreen{
        0.0f,
        0.0f
    };
};


HomographyResult calculateHomography(
    const std::array<cv::Point2f, 8>& imagePoints
)
{
    HomographyResult result;

    std::vector<cv::Point2f> modelPoints;

    modelPoints.reserve(8);

    for (const auto& p : config::kCTemplate) {

        modelPoints.emplace_back(
            static_cast<float>(p.x),
            static_cast<float>(p.y)
        );
    }

    const std::vector<cv::Point2f> image =
        toVector(imagePoints);

    // --------------------------------------------------------
    // 直接求 Homography。
    //
    // 對應關係已經固定，所以這裡不需要靠 RANSAC
    // 來猜哪個點是哪個點。
    // --------------------------------------------------------

    cv::Mat H = cv::findHomography(
        modelPoints,
        image,
        0
    );

    if (H.empty())
        return result;

    H.convertTo(
        result.H,
        CV_64F
    );

    // --------------------------------------------------------
    // Reprojection
    // --------------------------------------------------------

    std::vector<cv::Point2f> projected;

    cv::perspectiveTransform(
        modelPoints,
        projected,
        result.H
    );

    if (projected.size() != 8)
        return result;

    double sum = 0.0;

    for (int i = 0; i < 8; ++i) {

        const double dx =
            static_cast<double>(
                projected[i].x
            ) -
            imagePoints[i].x;

        const double dy =
            static_cast<double>(
                projected[i].y
            ) -
            imagePoints[i].y;

        sum += dx * dx + dy * dy;
    }

    result.rms =
        std::sqrt(sum / 8.0);

    if (!std::isfinite(result.rms))
        return result;

    // --------------------------------------------------------
    // Green prediction
    // --------------------------------------------------------

    cv::Point2f green;

    if (!projectHomographyPoint(
            result.H,
            config::kGreenCenterFromOrigin,
            green
        )) {
        return result;
    }

    result.expectedGreen = green;

    result.valid =
        result.rms <=
        config::kMaxHomographyRms;

    return result;
}


// ============================================================
// SQPnP evaluation
// ============================================================

struct PnPResult {

    bool valid = false;

    cv::Vec3d rvec{
        0.0,
        0.0,
        0.0
    };

    cv::Vec3d tvec{
        0.0,
        0.0,
        0.0
    };

    double rms = 0.0;

    cv::Point2f expectedGreen{
        0.0f,
        0.0f
    };
};


// ============================================================
// SQPnP
// ============================================================

PnPResult calculatePnP(
    const std::array<cv::Point2f, 8>& imagePoints
)
{
    PnPResult result;

    // --------------------------------------------------------
    // 3D model
    // --------------------------------------------------------

    const auto modelArray =
        buildCModel3D();

    std::vector<cv::Point3f> objectPoints(
        modelArray.begin(),
        modelArray.end()
    );

    const std::vector<cv::Point2f> imagePointsVector =
        toVector(imagePoints);

    // --------------------------------------------------------
    // Camera calibration
    //
    // 目前直接使用 config fallback。
    //
    // 若你的 pipeline 已經有 YAML calibration，
    // 下一步建議把 camera matrix / distortion 從 pipeline
    // 傳進來，而不是這裡固定 fallback。
    // --------------------------------------------------------

    const cv::Matx33d cameraMatrix =
        config::kFallbackCameraMatrix;

    cv::Mat distCoeffs(
        static_cast<int>(
            config::kFallbackDist.size()
        ),
        1,
        CV_64F
    );

    for (int i = 0;
         i < static_cast<int>(
                 config::kFallbackDist.size()
             );
         ++i) {

        distCoeffs.at<double>(i, 0) =
            config::kFallbackDist[i];
    }

    // --------------------------------------------------------
    // SQPnP
    // --------------------------------------------------------

    cv::Vec3d rvec;
    cv::Vec3d tvec;

    bool solved = false;

    try {

        solved = cv::solvePnP(
            objectPoints,
            imagePointsVector,
            cameraMatrix,
            distCoeffs,
            rvec,
            tvec,
            false,
            cv::SOLVEPNP_SQPNP
        );

    }
    catch (const cv::Exception&) {
        return result;
    }

    if (!solved)
        return result;

    // --------------------------------------------------------
    // Validate pose values
    // --------------------------------------------------------

    for (int i = 0; i < 3; ++i) {

        if (!std::isfinite(rvec[i]) ||
            !std::isfinite(tvec[i])) {

            return result;
        }
    }

    // --------------------------------------------------------
    // Reproject 8 points
    // --------------------------------------------------------

    std::vector<cv::Point2f> projected;

    try {

        cv::projectPoints(
            objectPoints,
            rvec,
            tvec,
            cameraMatrix,
            distCoeffs,
            projected
        );

    }
    catch (const cv::Exception&) {
        return result;
    }

    if (projected.size() != 8)
        return result;

    double sum = 0.0;

    for (int i = 0; i < 8; ++i) {

        const double dx =
            static_cast<double>(
                projected[i].x
            ) -
            imagePoints[i].x;

        const double dy =
            static_cast<double>(
                projected[i].y
            ) -
            imagePoints[i].y;

        sum += dx * dx + dy * dy;
    }

    result.rms =
        std::sqrt(sum / 8.0);

    if (!std::isfinite(result.rms))
        return result;

    // --------------------------------------------------------
    // Predict Green center
    // --------------------------------------------------------

    const cv::Point3f green =
        buildGreenModelPoint();

    std::vector<cv::Point3f> greenObject{
        green
    };

    std::vector<cv::Point2f> greenProjected;

    try {

        cv::projectPoints(
            greenObject,
            rvec,
            tvec,
            cameraMatrix,
            distCoeffs,
            greenProjected
        );

    }
    catch (const cv::Exception&) {
        return result;
    }

    if (greenProjected.size() != 1)
        return result;

    result.expectedGreen =
        greenProjected[0];

    result.rvec = rvec;
    result.tvec = tvec;

    result.valid =
        result.rms <=
        config::kMaxPnPRms;

    return result;
}


// ============================================================
// Validate geometric consistency
// ============================================================
//
// Homography 與 SQPnP 都應該描述同一個平面。
// 因此：
//
//       Green_H ≈ Green_PnP
//
// 如果差異非常大，代表：
//   - 點順序錯
//   - C 不是正確 C
//   - contour 失真太嚴重
//   - PnP 解錯
//   - calibration 不正確
//
// ============================================================

double greenPredictionError(
    const cv::Point2f& a,
    const cv::Point2f& b
)
{
    return cv::norm(a - b);
}


// ============================================================
// Find best ordering
//
// 共 16 種：
//   8 個 cyclic shift
//   × 2 個方向
//
// 對每一種：
//   1. Homography
//   2. SQPnP
//   3. Green prediction
//
// 最後選幾何一致性最好的結果。
//
// 這裡的 16 次不是在「找哪一個 C」，而是在解決
// contour 的 P0 index 不確定性。
// ============================================================

struct BestGeometry {

    bool found = false;

    std::array<cv::Point2f, 8> points{};

    HomographyResult homography;

    PnPResult pnp;

    double score =
        std::numeric_limits<double>::max();
};


BestGeometry findBestGeometry(
    const std::vector<cv::Point>& polygon
)
{
    BestGeometry best;

    for (int reverse = 0;
         reverse <= 1;
         ++reverse) {

        for (int offset = 0;
             offset < 8;
             ++offset) {

            const auto points =
                reorderPolygon(
                    polygon,
                    offset,
                    reverse != 0
                );

            const HomographyResult h =
                calculateHomography(points);

            const PnPResult p =
                calculatePnP(points);


            // =================================================
            // DEBUG
            // =================================================

            if (kDebugBlue) {

                std::cout
                    << "    [GEOMETRY]"
                    << " reverse="
                    << reverse
                    << " offset="
                    << offset
                    << " | H="
                    << (h.valid ? "OK" : "FAIL")
                    << " RMS="
                    << h.rms
                    << " | PnP="
                    << (p.valid ? "OK" : "FAIL")
                    << " RMS="
                    << p.rms;

                if (h.valid && p.valid) {

                    std::cout
                        << " | GreenΔ="
                        << greenPredictionError(
                               h.expectedGreen,
                               p.expectedGreen
                           );
                }

                std::cout
                    << '\n';
            }


            // ------------------------------------------------
            // 至少一個模型成功
            // ------------------------------------------------

            if (!h.valid && !p.valid) {

                if (kDebugBlue)
                    std::cout
                        << "        -> reject: H + PnP both fail\n";

                continue;
            }


            // ------------------------------------------------
            // Score
            // ------------------------------------------------

            double score = 0.0;

            if (h.valid) {

                score += h.rms;

            } else {

                score +=
                    config::kMaxHomographyRms * 5.0;
            }


            if (p.valid) {

                score += p.rms;

            } else {

                score +=
                    config::kMaxPnPRms * 5.0;
            }


            if (h.valid && p.valid) {

                const double greenError =
                    greenPredictionError(
                        h.expectedGreen,
                        p.expectedGreen
                    );

                score +=
                    greenError;

            }
            else {

                score +=
                    config::kMaxGreenPredictionError * 2.0;
            }


            if (kDebugBlue) {

                std::cout
                    << "        -> score="
                    << score;

                if (!best.found ||
                    score < best.score) {

                    std::cout
                        << " NEW BEST";
                }

                std::cout
                    << '\n';
            }


            if (!best.found ||
                score < best.score) {

                best.found = true;
                best.points = points;
                best.homography = h;
                best.pnp = p;
                best.score = score;
            }
        }
    }

    return best;
}

} // namespace


// ============================================================
// Main
// ============================================================

std::vector<BlueCandidate>
find_blue_candidates(
    const cv::Mat& blue_mask
)
{
    std::vector<BlueCandidate> candidates;

    if (blue_mask.empty()) {

        if (kDebugBlue)
            std::cout
                << "[BLUE] mask EMPTY\n";

        return candidates;
    }

    // ========================================================
    // Find contours
    // ========================================================

    std::vector<std::vector<cv::Point>> contours;

    cv::findContours(
        blue_mask,
        contours,
        cv::RETR_EXTERNAL,
        cv::CHAIN_APPROX_SIMPLE
    );

    if (kDebugBlue) {

        std::cout
            << "\n"
            << "============================================================\n"
            << "[BLUE] find_blue_candidates()\n"
            << "============================================================\n";

        std::cout
            << "[BLUE] mask size      : "
            << blue_mask.cols
            << " x "
            << blue_mask.rows
            << '\n';

        std::cout
            << "[BLUE] mask nonzero   : "
            << cv::countNonZero(blue_mask)
            << '\n';

        std::cout
            << "[BLUE] contours       : "
            << contours.size()
            << '\n';
    }

    if (contours.empty())
        return candidates;


    // ========================================================
    // Statistics
    // ========================================================

    int rejectArea = 0;
    int rejectPoints = 0;
    int rejectPerimeter = 0;
    int rejectApprox = 0;
    int rejectGeometry = 0;


    // ========================================================
    // Process every contour
    // ========================================================

    for (std::size_t contourIndex = 0;
         contourIndex < contours.size();
         ++contourIndex) {

        const auto& contour =
            contours[contourIndex];

        if (kDebugBlue) {

            std::cout
                << "\n"
                << "[BLUE][Contour "
                << contourIndex
                << "]\n";

            std::cout
                << "  contour points : "
                << contour.size()
                << '\n';
        }


        // ----------------------------------------------------
        // Area
        // ----------------------------------------------------

        const double area =
            cv::contourArea(contour);

        if (kDebugBlue) {

            std::cout
                << "  area           : "
                << area
                << '\n';

            std::cout
                << "  min area       : "
                << config::kMinBlueArea
                << '\n';
        }

        if (area <
            config::kMinBlueArea) {

            ++rejectArea;

            if (kDebugBlue)
                std::cout
                    << "  >>> REJECT: area\n";

            continue;
        }


        // ----------------------------------------------------
        // Contour point count
        // ----------------------------------------------------

        if (contour.size() <
            config::kMinContourPoints) {

            ++rejectPoints;

            if (kDebugBlue)
                std::cout
                    << "  >>> REJECT: contour point count\n"
                    << "      actual="
                    << contour.size()
                    << " required="
                    << config::kMinContourPoints
                    << '\n';

            continue;
        }


        // ----------------------------------------------------
        // Perimeter
        // ----------------------------------------------------

        const double perimeter =
            cv::arcLength(
                contour,
                true
            );

        if (kDebugBlue) {

            std::cout
                << "  perimeter      : "
                << perimeter
                << '\n';
        }

        if (perimeter <= 0.0) {

            ++rejectPerimeter;

            if (kDebugBlue)
                std::cout
                    << "  >>> REJECT: perimeter\n";

            continue;
        }


        // ----------------------------------------------------
        // Polygon approximation
        // ----------------------------------------------------

        std::vector<cv::Point> polygon;

        const double epsilon =
            config::kApproxEpsilonRatio *
            perimeter;

        cv::approxPolyDP(
            contour,
            polygon,
            epsilon,
            true
        );

        if (kDebugBlue) {

            std::cout
                << "  approx epsilon : "
                << epsilon
                << '\n';

            std::cout
                << "  approx points  : "
                << polygon.size()
                << '\n';

            std::cout
                << "  expected       : "
                << config::kExpectedPointCount
                << '\n';

            std::cout
                << "  polygon        : ";

            for (std::size_t i = 0;
                 i < polygon.size();
                 ++i) {

                std::cout
                    << "("
                    << polygon[i].x
                    << ","
                    << polygon[i].y
                    << ") ";

            }

            std::cout << '\n';
        }


        // ----------------------------------------------------
        // C 必須 8 點
        // ----------------------------------------------------

        if (polygon.size() !=
            config::kExpectedPointCount) {

            ++rejectApprox;

            if (kDebugBlue)
                std::cout
                    << "  >>> REJECT: polygon point count\n";

            continue;
        }


        // ====================================================
        // Geometry
        // ====================================================

        if (kDebugBlue)
            std::cout
                << "  >>> Geometry START\n";

        const BestGeometry geometry =
            findBestGeometry(
                polygon
            );

        if (!geometry.found) {

            ++rejectGeometry;

            if (kDebugBlue)
                std::cout
                    << "  >>> REJECT: geometry\n";

            continue;
        }


        // ====================================================
        // Geometry debug
        // ====================================================

        if (kDebugBlue) {

            std::cout
                << "  >>> Geometry FOUND\n";

            std::cout
                << "      score       : "
                << geometry.score
                << '\n';

            std::cout
                << "      H valid     : "
                << geometry.homography.valid
                << '\n';

            std::cout
                << "      H RMS       : "
                << geometry.homography.rms
                << '\n';

            std::cout
                << "      H Green     : ("
                << geometry.homography.expectedGreen.x
                << ", "
                << geometry.homography.expectedGreen.y
                << ")\n";

            std::cout
                << "      PnP valid   : "
                << geometry.pnp.valid
                << '\n';

            std::cout
                << "      PnP RMS     : "
                << geometry.pnp.rms
                << '\n';

            std::cout
                << "      PnP Green   : ("
                << geometry.pnp.expectedGreen.x
                << ", "
                << geometry.pnp.expectedGreen.y
                << ")\n";

            if (geometry.homography.valid &&
                geometry.pnp.valid) {

                std::cout
                    << "      H/P Green Δ : "
                    << greenPredictionError(
                           geometry.homography.expectedGreen,
                           geometry.pnp.expectedGreen
                       )
                    << " px\n";
            }

            std::cout
                << "      H threshold : "
                << config::kMaxHomographyRms
                << '\n';

            std::cout
                << "      PnP thresh  : "
                << config::kMaxPnPRms
                << '\n';

            std::cout
                << "      Green thresh: "
                << config::kMaxGreenPredictionError
                << '\n';

            std::cout
                << "      Selected points:\n";

            for (int i = 0; i < 8; ++i) {

                std::cout
                    << "        ["
                    << i
                    << "] "
                    << geometry.points[i].x
                    << ", "
                    << geometry.points[i].y
                    << '\n';
            }
        }


        // ====================================================
        // Build candidate
        // ====================================================

        BlueCandidate candidate;

        candidate.contour =
            contour;

        candidate.area =
            area;

        candidate.points =
            geometry.points;


        // ====================================================
        // Homography
        // ====================================================

        candidate.homography =
            geometry.homography.H;

        candidate.expected_green_h =
            geometry.homography.expectedGreen;

        candidate.homography_reprojection_rms =
            geometry.homography.rms;

        candidate.homography_valid =
            geometry.homography.valid;


        // ====================================================
        // SQPnP
        // ====================================================

        candidate.rvec =
            geometry.pnp.rvec;

        candidate.tvec =
            geometry.pnp.tvec;

        candidate.expected_green_pnp =
            geometry.pnp.expectedGreen;

        candidate.pnp_reprojection_rms =
            geometry.pnp.rms;

        candidate.pnp_valid =
            geometry.pnp.valid;


        // ====================================================
        // Green prediction consistency
        // ====================================================

        if (candidate.homography_valid &&
            candidate.pnp_valid) {

            candidate.green_prediction_error =
                greenPredictionError(
                    candidate.expected_green_h,
                    candidate.expected_green_pnp
                );

        }
        else {

            candidate.green_prediction_error =
                std::numeric_limits<double>::max();
        }


        // ====================================================
        // Final validation
        // ====================================================

        candidate.valid =
            candidate.homography_valid &&
            candidate.pnp_valid &&
            candidate.green_prediction_error <=
                config::kMaxGreenPredictionError;


        if (kDebugBlue) {

            std::cout
                << "  --------------------------------------------------\n";

            std::cout
                << "  FINAL C RESULT\n";

            std::cout
                << "    H valid      : "
                << candidate.homography_valid
                << '\n';

            std::cout
                << "    H RMS        : "
                << candidate.homography_reprojection_rms
                << '\n';

            std::cout
                << "    PnP valid    : "
                << candidate.pnp_valid
                << '\n';

            std::cout
                << "    PnP RMS      : "
                << candidate.pnp_reprojection_rms
                << '\n';

            std::cout
                << "    Green Δ      : "
                << candidate.green_prediction_error
                << '\n';

            std::cout
                << "    FINAL VALID  : "
                << candidate.valid
                << '\n';

            if (!candidate.valid) {

                std::cout
                    << "    >>> CANDIDATE INVALID\n";
            }
        }


        candidates.push_back(
            std::move(candidate)
        );
    }


    // ========================================================
    // Summary
    // ========================================================

    if (kDebugBlue) {

        std::cout
            << "\n"
            << "============================================================\n"
            << "[BLUE] SUMMARY\n"
            << "============================================================\n";

        std::cout
            << "  contours              : "
            << contours.size()
            << '\n';

        std::cout
            << "  reject area           : "
            << rejectArea
            << '\n';

        std::cout
            << "  reject contour points : "
            << rejectPoints
            << '\n';

        std::cout
            << "  reject perimeter      : "
            << rejectPerimeter
            << '\n';

        std::cout
            << "  reject approx != 8    : "
            << rejectApprox
            << '\n';

        std::cout
            << "  reject geometry       : "
            << rejectGeometry
            << '\n';

        std::cout
            << "  candidates            : "
            << candidates.size()
            << '\n';

        int validCount = 0;

        for (const auto& c : candidates) {

            if (c.valid)
                ++validCount;
        }

        std::cout
            << "  valid candidates      : "
            << validCount
            << '\n';

        std::cout
            << "============================================================\n";
    }

    return candidates;
}

} // namespace tel::core