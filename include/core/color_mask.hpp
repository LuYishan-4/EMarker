// ============================================================
// color_mask.hpp
//
// 職責：BGR frame -> HSV -> inRange -> morphology mask
//
// 效率重點：
//   - HSV 轉換只做一次，green/blue mask 共用同一張 HSV 影像
//     (取代 Python 版分別對 green/blue 各呼叫一次 cvtColor)
//   - morphology kernel 只建立一次 (static, 不隨 frame 重建)
// ============================================================
#pragma once

#include <opencv2/core.hpp>
#include "core/config.hpp"

namespace tel::core {

// 將 frame 轉為 HSV，供 green/blue mask 共用
cv::Mat to_hsv(const cv::Mat& bgr_frame);

// 依據 HsvRange 產生二值遮罩 (含 open+close 形態學)
cv::Mat make_mask(const cv::Mat& hsv, const config::HsvRange& range);

} // namespace tel::core