#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "core/types.hpp"

namespace tel::core {

// ============================================================
// 找出影像中的所有 C
//
// 每個 BlueCandidate 包含：
//   - C 的 8 個影像座標
//   - 原始 our
//   - 面積
//   - Homography
//   - SQPnP pose
//   - 8 點 reprojection error
//   - 理論 Green center
//   - Green prediction 驗證
//
// 注意：
//   這裡只負責「C 候選偵測與幾何驗證」。
//   不負責和實際 Green contour 做匹配。
// ============================================================

std::vector<BlueCandidate> find_blue_candidates(const cv::Mat &blue_mask);

} // namespace tel::core
