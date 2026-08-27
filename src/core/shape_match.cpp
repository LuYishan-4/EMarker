#include "core/shape_match.hpp"
#include "core/config.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>

namespace tel::core {

namespace {

constexpr std::size_t kPointCount = 8;

// ============================================================
// PointStats
// ============================================================

struct PointStats {
    cv::Point2d mean{0.0, 0.0};
    std::vector<cv::Point2d> centered;
    double norm = 0.0;
};

// ============================================================
// array -> vector
//
// 只有數學計算內部使用。
// ============================================================

std::vector<cv::Point2f> to_vector(
    const std::array<cv::Point2f, kPointCount>& points
) {
    return std::vector<cv::Point2f>(
        points.begin(),
        points.end()
    );
}

// ============================================================
// compute stats
// ============================================================

PointStats compute_stats(
    const std::vector<cv::Point2f>& pts
) {
    PointStats s;

    if (pts.empty())
        return s;

    s.centered.resize(pts.size());

    cv::Point2d sum{0.0, 0.0};

    for (const auto& p : pts) {
        sum.x += p.x;
        sum.y += p.y;
    }

    s.mean =
        sum *
        (1.0 / static_cast<double>(pts.size()));

    double sq_sum = 0.0;

    for (std::size_t i = 0;
         i < pts.size();
         ++i) {

        s.centered[i] =
            cv::Point2d(
                pts[i].x,
                pts[i].y
            ) -
            s.mean;

        sq_sum +=
            s.centered[i].x *
            s.centered[i].x +
            s.centered[i].y *
            s.centered[i].y;
    }

    s.norm = std::sqrt(sq_sum);

    return s;
}

// ============================================================
// Similarity solve
//
// template -> image
//
// row-vector convention:
//     dst = scale * (src @ R) + t
// ============================================================

std::optional<SimilarityResult>
solve_similarity(
    const std::vector<cv::Point2d>& src_raw,
    const PointStats& src_stats,
    const std::vector<cv::Point2f>& dst,
    const PointStats& dst_stats
) {
    const std::size_t n =
        src_raw.size();

    if (n != kPointCount)
        return std::nullopt;

    if (dst.size() != kPointCount)
        return std::nullopt;

    if (src_stats.norm < 1e-9 ||
        dst_stats.norm < 1e-9) {

        return std::nullopt;
    }

    // --------------------------------------------------------
    // H = src_n^T * dst_n
    // --------------------------------------------------------

    double h00 = 0.0;
    double h01 = 0.0;
    double h10 = 0.0;
    double h11 = 0.0;

    for (std::size_t i = 0;
         i < n;
         ++i) {

        const double sx =
            src_stats.centered[i].x /
            src_stats.norm;

        const double sy =
            src_stats.centered[i].y /
            src_stats.norm;

        const double dx =
            dst_stats.centered[i].x /
            dst_stats.norm;

        const double dy =
            dst_stats.centered[i].y /
            dst_stats.norm;

        h00 += sx * dx;
        h01 += sx * dy;
        h10 += sy * dx;
        h11 += sy * dy;
    }

    const cv::Matx22d H(
        h00, h01,
        h10, h11
    );

    cv::Matx22d U;
    cv::Matx22d Vt;
    cv::Matx21d W;

    cv::SVD::compute(
        H,
        W,
        U,
        Vt,
        cv::SVD::FULL_UV
    );

    cv::Matx22d R =
        Vt.t() * U.t();

    // --------------------------------------------------------
    // 保證 det(R) = +1
    // --------------------------------------------------------

    if (cv::determinant(R) < 0.0) {

        cv::Matx22d Vt_fixed = Vt;

        Vt_fixed(1, 0) *= -1.0;
        Vt_fixed(1, 1) *= -1.0;

        R =
            Vt_fixed.t() *
            U.t();
    }

    // --------------------------------------------------------
    // Scale
    // --------------------------------------------------------

    const double scale =
        dst_stats.norm /
        src_stats.norm;

    // --------------------------------------------------------
    // Translation
    // --------------------------------------------------------

    const cv::Matx21d src_mean_vec(
        src_stats.mean.x,
        src_stats.mean.y
    );

    const cv::Matx21d rotated_src_mean =
        R.t() *
        src_mean_vec;

    const cv::Point2d translation(
        dst_stats.mean.x -
            scale * rotated_src_mean(0),

        dst_stats.mean.y -
            scale * rotated_src_mean(1)
    );

    // --------------------------------------------------------
    // Result
    // --------------------------------------------------------

    SimilarityResult result;

    result.scale = scale;

    result.rotation_deg =
        std::atan2(
            R(0, 1),
            R(0, 0)
        ) *
        180.0 /
        CV_PI;

    result.translation =
        cv::Point2f(
            static_cast<float>(
                translation.x
            ),
            static_cast<float>(
                translation.y
            )
        );

    double sq_err_sum = 0.0;

    for (std::size_t i = 0;
         i < kPointCount;
         ++i) {

        const cv::Matx21d src_vec(
            src_raw[i].x,
            src_raw[i].y
        );

        const cv::Matx21d rotated =
            R.t() *
            src_vec;

        const double px =
            scale * rotated(0) +
            translation.x;

        const double py =
            scale * rotated(1) +
            translation.y;

        result.predicted[i] =
            cv::Point2f(
                static_cast<float>(px),
                static_cast<float>(py)
            );

        const double ex =
            px - dst[i].x;

        const double ey =
            py - dst[i].y;

        sq_err_sum +=
            ex * ex +
            ey * ey;
    }

    result.rms_error =
        std::sqrt(
            sq_err_sum /
            static_cast<double>(
                kPointCount
            )
        );

    return result;
}

// ============================================================
// Template cache
// ============================================================

struct TemplateCache {

    std::vector<cv::Point2d> raw;

    PointStats stats;
};

const TemplateCache&
template_cache() {

    static const TemplateCache cache =
        [] {

            TemplateCache c;

            c.raw.reserve(kPointCount);

            for (const auto& p :
                 config::kCTemplate) {

                c.raw.emplace_back(
                    p.x,
                    p.y
                );
            }

            std::vector<cv::Point2f> raw_f;

            raw_f.reserve(kPointCount);

            for (const auto& p :
                 config::kCTemplate) {

                raw_f.emplace_back(
                    static_cast<float>(p.x),
                    static_cast<float>(p.y)
                );
            }

            c.stats =
                compute_stats(raw_f);

            return c;
        }();

    return cache;
}

} // namespace


// ============================================================
// order_polygon
//
// 一般 polygon ordering。
// ============================================================

std::vector<cv::Point2f>
order_polygon(
    const std::vector<cv::Point2f>& points
) {
    if (points.empty())
        return {};

    cv::Point2f center(
        0.0f,
        0.0f
    );

    for (const auto& p :
         points) {

        center += p;
    }

    center *=
        1.0f /
        static_cast<float>(
            points.size()
        );

    std::vector<std::size_t> idx(
        points.size()
    );

    std::iota(
        idx.begin(),
        idx.end(),
        0
    );

    std::sort(
        idx.begin(),
        idx.end(),
        [&](std::size_t a,
            std::size_t b) {

            const double angle_a =
                std::atan2(
                    points[a].y - center.y,
                    points[a].x - center.x
                );

            const double angle_b =
                std::atan2(
                    points[b].y - center.y,
                    points[b].x - center.x
                );

            return angle_a < angle_b;
        }
    );

    std::vector<cv::Point2f> ordered;

    ordered.reserve(points.size());

    for (const auto i :
         idx) {

        ordered.push_back(
            points[i]
        );
    }

    return ordered;
}


// ============================================================
// cyclic variants
//
// C 固定 8 點。
// 不再使用 vector 儲存 variant。
// ============================================================

std::vector<std::array<cv::Point2f, 8>>
cyclic_variants(
    const std::array<cv::Point2f, 8>& points
) {
    std::vector<
        std::array<cv::Point2f, 8>
    > variants;

    variants.reserve(16);

    for (int shift = 0;
         shift < 8;
         ++shift) {

        // ----------------------------------------------------
        // 正方向
        // ----------------------------------------------------

        std::array<cv::Point2f, 8>
            normal{};

        for (int i = 0;
             i < 8;
             ++i) {

            normal[i] =
                points[
                    (i + shift) % 8
                ];
        }

        variants.push_back(
            normal
        );

        // ----------------------------------------------------
        // 反方向
        // ----------------------------------------------------

        std::array<cv::Point2f, 8>
            reversed{};

        for (int i = 0;
             i < 8;
             ++i) {

            reversed[i] =
                points[
                    (shift - i + 8) % 8
                ];
        }

        variants.push_back(
            reversed
        );
    }

    return variants;
}


// ============================================================
// estimate similarity
// ============================================================

std::optional<SimilarityResult>
estimate_similarity(
    const std::vector<cv::Point2f>& src,
    const std::vector<cv::Point2f>& dst
) {
    if (src.size() != kPointCount ||
        dst.size() != kPointCount) {

        return std::nullopt;
    }

    std::vector<cv::Point2d>
        src_raw;

    src_raw.reserve(kPointCount);

    for (const auto& p :
         src) {

        src_raw.emplace_back(
            p.x,
            p.y
        );
    }

    const PointStats src_stats =
        compute_stats(src);

    const PointStats dst_stats =
        compute_stats(dst);

    return solve_similarity(
        src_raw,
        src_stats,
        dst,
        dst_stats
    );
}


// ============================================================
// estimate similarity from C template
// ============================================================

std::optional<SimilarityResult>
estimate_similarity_from_template(
    const std::array<cv::Point2f, 8>& dst
) {
    const TemplateCache& tc =
        template_cache();

    const std::vector<cv::Point2f>
        dst_vector =
            to_vector(dst);

    const PointStats dst_stats =
        compute_stats(dst_vector);

    return solve_similarity(
        tc.raw,
        tc.stats,
        dst_vector,
        dst_stats
    );
}


// ============================================================
// Match C
//
// 現在不是「只找一個候選」的 detector。
// 這裡仍保留原本 API：
//   從全部候選中找 shape 最好的 C。
//
// Pipeline 若需要所有 C，直接使用
// find_blue_candidates() 回傳的 vector。
// ============================================================

CMatch match_c_shape(
    const std::vector<BlueCandidate>& candidates
) {
    CMatch best;

    double best_error =
        std::numeric_limits<double>::max();

    for (const auto& candidate :
         candidates) {

        for (const auto& variant :
             cyclic_variants(
                 candidate.points
             )) {

            const auto result =
                estimate_similarity_from_template(
                    variant
                );

            if (!result.has_value())
                continue;

            if (result->scale <
                    config::kMinScale ||
                result->scale >
                    config::kMaxScale) {

                continue;
            }

            if (result->rms_error >
                config::kMaxShapeError) {

                continue;
            }

            if (result->rms_error >=
                best_error) {

                continue;
            }

            best_error =
                result->rms_error;

            best.found = true;

            best.points =
                variant;

            best.predicted =
                result->predicted;

            best.shape_error =
                result->rms_error;

            best.scale =
                result->scale;

            best.rotation_deg =
                result->rotation_deg;

            best.translation =
                result->translation;

            best.contour =
                candidate.contour;

            best.area =
                candidate.area;
        }
    }

    return best;
}


// ============================================================
// Validate Green anchor
//
// 這個函數保留原本 similarity transform 的 Green 預測。
// ============================================================

GreenAnchorValidation
validate_green_anchor(
    const CMatch& match,
    const std::optional<cv::Point2f>& green_center
) {
    GreenAnchorValidation v;

    if (!match.found ||
        !green_center.has_value()) {

        return v;
    }

    v.has_reference = true;

    const double angle =
        match.rotation_deg *
        CV_PI /
        180.0;

    const double cos_a =
        std::cos(angle);

    const double sin_a =
        std::sin(angle);

    const auto& g =
        config::kGreenCenterFromOrigin;

    const double rx =
        g.x * cos_a +
        g.y * sin_a;

    const double ry =
        -g.x * sin_a +
        g.y * cos_a;

    const double px =
        match.translation.x +
        match.scale * rx;

    const double py =
        match.translation.y +
        match.scale * ry;

    v.predicted =
        cv::Point2f(
            static_cast<float>(px),
            static_cast<float>(py)
        );

    v.error =
        cv::norm(
            v.predicted -
            *green_center
        );

    v.valid =
        v.error <=
        config::kMaxGreenAnchorError;

    return v;
}

} // namespace tel::core