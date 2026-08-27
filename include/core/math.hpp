#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>

namespace tel::math {

// ============================================================
// Line
//
// A*x + B*y = C
// ============================================================

struct Line {
    double A = 0.0;
    double B = 0.0;
    double C = 0.0;

    bool valid = false;

    // --------------------------------------------------------
    // Least-squares line fitting
    //
    // 使用 PCA / covariance eigenvector。
    // 對接近垂直的線也穩定。
    // --------------------------------------------------------

    static Line fitLeastSquares(
        const std::vector<cv::Point>& points
    ) {
        Line result{};

        if (points.size() < 2)
            return result;

        double meanX = 0.0;
        double meanY = 0.0;

        for (const auto& p : points) {
            meanX += p.x;
            meanY += p.y;
        }

        meanX /= static_cast<double>(points.size());
        meanY /= static_cast<double>(points.size());

        double xx = 0.0;
        double xy = 0.0;
        double yy = 0.0;

        for (const auto& p : points) {
            const double dx =
                static_cast<double>(p.x) - meanX;

            const double dy =
                static_cast<double>(p.y) - meanY;

            xx += dx * dx;
            xy += dx * dy;
            yy += dy * dy;
        }

        // ----------------------------------------------------
        // Covariance matrix:
        //
        // [ xx xy ]
        // [ xy yy ]
        //
        // 找最大 eigenvalue 對應方向。
        // ----------------------------------------------------

        const double trace = xx + yy;

        const double diff = xx - yy;

        const double discriminant =
            std::sqrt(
                diff * diff +
                4.0 * xy * xy
            );

        const double lambda =
            0.5 * (trace + discriminant);

        double dirX = 1.0;
        double dirY = 0.0;

        if (std::abs(xy) > 1e-12 ||
            std::abs(lambda - xx) > 1e-12) {

            dirX = xy;
            dirY = lambda - xx;

            const double len =
                std::hypot(dirX, dirY);

            if (len > 1e-12) {
                dirX /= len;
                dirY /= len;
            } else {
                dirX = 1.0;
                dirY = 0.0;
            }
        } else {
            if (yy > xx) {
                dirX = 0.0;
                dirY = 1.0;
            }
        }

        // ----------------------------------------------------
        // Direction vector = (dirX, dirY)
        //
        // Normal vector = (-dirY, dirX)
        // ----------------------------------------------------

        result.A = -dirY;
        result.B = dirX;

        result.C =
            result.A * meanX +
            result.B * meanY;

        result.valid = true;

        return result;
    }

    // --------------------------------------------------------
    // Intersection of two lines
    // --------------------------------------------------------

    static std::optional<cv::Point2f> intersect(
        const Line& l1,
        const Line& l2
    ) {
        if (!l1.valid || !l2.valid)
            return std::nullopt;

        const double determinant =
            l1.A * l2.B -
            l2.A * l1.B;

        // 平行 / 幾乎平行
        if (std::abs(determinant) < 1e-10)
            return std::nullopt;

        const double x =
            (
                l1.C * l2.B -
                l2.C * l1.B
            ) / determinant;

        const double y =
            (
                l1.A * l2.C -
                l2.A * l1.C
            ) / determinant;

        if (!std::isfinite(x) ||
            !std::isfinite(y)) {
            return std::nullopt;
        }

        return cv::Point2f(
            static_cast<float>(x),
            static_cast<float>(y)
        );
    }
};


// ============================================================
// Quad
//
// 四個點依照 contour 順序排列：
//
// 0 → 1
// ↑   ↓
// 3 ← 2
//
// 也就是相鄰點就是一條邊。
// ============================================================

struct Quad {
    std::array<cv::Point2f, 4> points{};

    cv::Point2f& operator[](
        std::size_t index
    ) {
        return points[index];
    }

    const cv::Point2f& operator[](
        std::size_t index
    ) const {
        return points[index];
    }

    // --------------------------------------------------------
    // Polygon area
    //
    // Shoelace formula
    // --------------------------------------------------------

    double area() const {
        double sum = 0.0;

        for (int i = 0; i < 4; ++i) {
            const int j = (i + 1) % 4;

            sum +=
                static_cast<double>(points[i].x) *
                points[j].y;

            sum -=
                static_cast<double>(points[j].x) *
                points[i].y;
        }

        return std::abs(sum) * 0.5;
    }

    // --------------------------------------------------------
    // Perimeter
    // --------------------------------------------------------

    double perimeter() const {
        double result = 0.0;

        for (int i = 0; i < 4; ++i) {
            const int j = (i + 1) % 4;

            result +=
                cv::norm(
                    points[j] -
                    points[i]
                );
        }

        return result;
    }
};

} // namespace tel::math