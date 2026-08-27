#include "pipeline/emarker.hpp"

#include "core/green_anchor.hpp"
#include "core/blue_candidates.hpp"
#include "core/config.hpp"

#include <cmath>
#include <limits>
#include <utility>
#include <vector>
#include <iostream>

namespace tel::pipeline {

CMarkerPipeline::CMarkerPipeline(
    const cv::Matx33d& camera_matrix,
    const cv::Mat& dist_coeffs)
    : camera_matrix_(camera_matrix),
      dist_coeffs_(dist_coeffs.clone()) {}

std::vector<CMarker> CMarkerPipeline::process(
    const cv::Mat& blue_binary,
    const cv::Mat& green_binary) {

    std::vector<CMarker> markers;
    
    if (blue_binary.empty() || green_binary.empty())
        return markers;

    // ============================================================
    // 1. 找出所有 Green
    // ============================================================

    const auto green_result =
        tel::core::find_green_anchors(green_binary);

    if (!green_result.has_value())
        return markers;

    // ============================================================
    // 2. 找出所有 C
    //
    // blue_candidates.cpp 已經計算：
    //
    //   - 8 個 C 點
    //   - Homography
    //   - SQPnP
    //   - expected_green_h
    //   - expected_green_pnp
    //   - reprojection RMS
    //
    // Pipeline 不重新計算理論 Green。
    // ============================================================

    auto blue_candidates =
        tel::core::find_blue_candidates(blue_binary);

    if (blue_candidates.empty())
        return markers;

    // ============================================================
    // 3. Green 使用狀態
    //
    // 一個 Green 只能配一個 C。
    // ============================================================

    std::vector<bool> green_used(
        green_result->centers.size(),
        false
    );

    // ============================================================
    // 4. 逐一處理 C
    // ============================================================

    for (auto& candidate : blue_candidates) {

        // --------------------------------------------------------
        // C 本身的 H / PnP 必須先通過
        // --------------------------------------------------------

        if (!candidate.homography_valid)
            continue;

        if (!candidate.pnp_valid)
            continue;

        // --------------------------------------------------------
        // H / PnP 預測 Green
        // --------------------------------------------------------

        const cv::Point2f& green_h =
            candidate.expected_green_h;

        const cv::Point2f& green_pnp =
            candidate.expected_green_pnp;

        if (!std::isfinite(green_h.x) ||
            !std::isfinite(green_h.y) ||
            !std::isfinite(green_pnp.x) ||
            !std::isfinite(green_pnp.y)) {
            continue;
        }

        // --------------------------------------------------------
        // Homography 與 SQPnP 預測的 Green 必須一致
        // --------------------------------------------------------

        const double hp_error =
            cv::norm(green_h - green_pnp);

        if (hp_error >
            config::kMaxGreenPredictionError) {
            continue;
        }

        // --------------------------------------------------------
        // 理論 Green center
        //
        // H + PnP 兩種結果取平均。
        // --------------------------------------------------------

        const cv::Point2f expected_green(
            (green_h.x + green_pnp.x) * 0.5f,
            (green_h.y + green_pnp.y) * 0.5f
        );

        // --------------------------------------------------------
        // 5. 找最接近的實際 Green
        // --------------------------------------------------------

        int best_green_index = -1;

        double best_error =
            std::numeric_limits<double>::max();

        for (std::size_t i = 0;
             i < green_result->centers.size();
             ++i) {

            if (green_used[i])
                continue;

            // ----------------------------------------------------
            // 保留原本 Green 幾何驗證
            // ----------------------------------------------------

            const float area_ratio =
                green_result->area_ratios[i];

            if (area_ratio < 0.7f)
                continue;

            // ----------------------------------------------------
            // 實際 Green center
            // ----------------------------------------------------

            const cv::Point2f& actual_green =
                green_result->centers[i];

            const double error =
                cv::norm(
                    expected_green -
                    actual_green
                );

            if (error < best_error) {
                best_error = error;
                best_green_index =
                    static_cast<int>(i);
            }
        }

        // --------------------------------------------------------
        // 沒找到對應 Green
        // --------------------------------------------------------

        if (best_green_index < 0)
            continue;

        // --------------------------------------------------------
        // Green prediction 必須在允許誤差內
        // --------------------------------------------------------

        if (best_error >
            config::kMaxGreenPredictionError) {
            continue;
        }

        // --------------------------------------------------------
        // 成功配對
        // --------------------------------------------------------

        green_used[
            static_cast<std::size_t>(best_green_index)
        ] = true;

        candidate.green_prediction_error =
            best_error;

        candidate.valid = true;

        // ========================================================
        // 6. 建立 CMarker
        //
        // 注意：
        // 這裡使用 candidate 已經算好的資料。
        // 不重新做任何幾何/PnP/Homography。
        // ========================================================

        const cv::Point2f& actual_green =
            green_result->centers[
                static_cast<std::size_t>(best_green_index)
            ];

        markers.emplace_back(
            candidate,
            actual_green,
            expected_green
        );
    }

    return markers;
}

} // namespace tel::pipeline