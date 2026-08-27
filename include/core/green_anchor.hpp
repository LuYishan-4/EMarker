#pragma once

#include "core/math.hpp"

#include <array>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>

namespace tel::core {

// ============================================================
// GreenAnchorResult
// ============================================================
//
// 所有偵測到的 Green Anchor。
// 同一個 index 對應同一個 Anchor：
//
// centers[i]
// areas[i]
// area_ratios[i]
// quads[i]
//
// ============================================================

struct GreenAnchorResult {

    // --------------------------------------------------------
    // 綠色 contour 的質心
    // --------------------------------------------------------

    std::vector<cv::Point2f> centers;

    // --------------------------------------------------------
    // 實際 contour 面積
    // 單位：pixel²
    // --------------------------------------------------------

    std::vector<float> areas;

    // --------------------------------------------------------
    // 面積比例
    //
    // actual contour area
    // -------------------
    // recovered sharp quad area
    //
    // 1.0 → 幾乎沒有圓角
    // <1  → 有面積被圓角吃掉
    // --------------------------------------------------------

    std::vector<float> area_ratios;

    // --------------------------------------------------------
    // RANSAC 還原出的原始尖角 Quad
    // --------------------------------------------------------

    std::vector<math::Quad> quads;
};


// ============================================================
// Find all green anchors
// ============================================================

std::optional<GreenAnchorResult>
find_green_anchors(
    const cv::Mat& green_mask
);

} // namespace tel::core