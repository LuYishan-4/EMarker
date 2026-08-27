#include "core/green_anchor.hpp"
#include "core/config.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace tel::core {

std::optional<GreenAnchorResult>
find_green_anchors(
    const cv::Mat& green_mask
) {
    std::cout
        << "\n============================================================\n"
        << "[GREEN] find_green_anchors()\n"
        << "============================================================\n";

    if (green_mask.empty()) {

        std::cout
            << "[GREEN] ERROR: mask is empty\n";

        return std::nullopt;
    }

    std::cout
        << "[GREEN] mask size    : "
        << green_mask.cols
        << " x "
        << green_mask.rows
        << '\n';

    const int nonzero =
        cv::countNonZero(green_mask);

    std::cout
        << "[GREEN] nonzero     : "
        << nonzero
        << '\n';

    if (nonzero == 0) {

        std::cout
            << "[GREEN] >>> NO GREEN PIXELS\n";

        return std::nullopt;
    }

    // ========================================================
    // Find contours
    // ========================================================

    std::vector<std::vector<cv::Point>> contours;

    cv::findContours(
        green_mask,
        contours,
        cv::RETR_EXTERNAL,
        cv::CHAIN_APPROX_SIMPLE
    );

    std::cout
        << "[GREEN] contours    : "
        << contours.size()
        << '\n';

    if (contours.empty()) {

        std::cout
            << "[GREEN] >>> NO CONTOURS\n";

        return std::nullopt;
    }

    GreenAnchorResult result;

    // ========================================================
    // Process contours
    // ========================================================

    for (std::size_t i = 0;
         i < contours.size();
         ++i) {

        const auto& contour =
            contours[i];

        std::cout
            << "\n[GREEN][Contour "
            << i
            << "]\n";

        std::cout
            << "  points      : "
            << contour.size()
            << '\n';

        // ----------------------------------------------------
        // Area
        // ----------------------------------------------------

        const double area =
            std::abs(
                cv::contourArea(contour)
            );

        std::cout
            << "  area        : "
            << area
            << '\n';

        std::cout
            << "  min area    : "
            << config::kMinGreenArea
            << '\n';

        if (area <=
            config::kMinGreenArea) {

            std::cout
                << "  >>> REJECT: area\n";

            continue;
        }

        // ----------------------------------------------------
        // Bounding box
        // ----------------------------------------------------

        const cv::Rect bbox =
            cv::boundingRect(contour);

        std::cout
            << "  bbox        : "
            << bbox.width
            << " x "
            << bbox.height
            << '\n';

        // ----------------------------------------------------
        // Center
        // ----------------------------------------------------

        const cv::Moments moments =
            cv::moments(contour);

        std::cout
            << "  m00         : "
            << moments.m00
            << '\n';

        if (std::abs(moments.m00) < 1e-9) {

            std::cout
                << "  >>> REJECT: m00 == 0\n";

            continue;
        }

        const cv::Point2f center(
            static_cast<float>(
                moments.m10 / moments.m00
            ),
            static_cast<float>(
                moments.m01 / moments.m00
            )
        );

        std::cout
            << "  center      : ("
            << center.x
            << ", "
            << center.y
            << ")\n";

        // ----------------------------------------------------
        // minAreaRect
        //
        // 只做 DEBUG。
        // 不再用它決定 Green 是否有效。
        // ----------------------------------------------------

        const cv::RotatedRect rect =
            cv::minAreaRect(contour);

        std::cout
            << "  minAreaRect:\n"
            << "    center = ("
            << rect.center.x
            << ", "
            << rect.center.y
            << ")\n"
            << "    size   = ("
            << rect.size.width
            << ", "
            << rect.size.height
            << ")\n"
            << "    angle  = "
            << rect.angle
            << '\n';

        cv::Point2f rectPoints[4];

        rect.points(rectPoints);

        for (int p = 0; p < 4; ++p) {

            std::cout
                << "    P"
                << p
                << " = ("
                << rectPoints[p].x
                << ", "
                << rectPoints[p].y
                << ")\n";
        }

        // ----------------------------------------------------
        // Accepted
        // ----------------------------------------------------

        result.centers.push_back(center);

        result.areas.push_back(
            static_cast<float>(area)
        );

        // ----------------------------------------------------
        // 保留原 API
        //
        // 這裡不再依賴 recoverQuad。
        // 如果其他地方只是拿 area_ratio 做 debug，
        // 對 Green 來說可以直接設成 1。
        // ----------------------------------------------------

        result.area_ratios.push_back(1.0f);

        // ----------------------------------------------------
        // Quad
        //
        // 使用 minAreaRect 作為 debug / compatibility。
        // ----------------------------------------------------

        math::Quad quad{};

        for (int p = 0; p < 4; ++p) {

            quad[p] =
                rectPoints[p];
        }

        result.quads.push_back(quad);

        std::cout
            << "  >>> ACCEPT GREEN\n";
    }

    // ========================================================
    // Result
    // ========================================================

    if (result.centers.empty()) {

        std::cout
            << "\n[GREEN] >>> NO VALID GREEN ANCHOR\n";

        return std::nullopt;
    }

    std::cout
        << "\n[GREEN] >>> ACCEPTED: "
        << result.centers.size()
        << " Green anchor(s)\n";

    return result;
}

} // namespace tel::core