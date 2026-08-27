// ============================================================
// shape_match.hpp
//
// 職責：
//   1. 8 點多邊形依角度排序 (order_polygon)
//   2. 產生 16 種 cyclic 起點/方向變體 (cyclic_variants)
//   3. Kabsch/Umeyama 相似度變換估計 (C_TEMPLATE -> image)
//   4. 從候選中挑出誤差最小且合法的 C 形狀 (match_c_shape)
//   5. 綠色 anchor 反投影驗證 (validate_green_anchor)
//
// 效率重點：
//   - src 端固定為 C_TEMPLATE，其 mean/centered/norm 只計算一次
//     (static，lazy-init)，取代 Python 版每次呼叫都重算 src 統計量。
//   - estimate_similarity_from_template() 是實際被 match_c_shape()
//     高頻呼叫的路徑；estimate_similarity() 保留通用版本供其他用途
//     (例如未來要比對別的模板) 或單元測試使用。
// ============================================================
#pragma once

#include <opencv2/core.hpp>
#include <optional>
#include <vector>
#include "core/types.hpp"

namespace tel::core {

// 依幾何中心角度排序 8 點
std::vector<cv::Point2f> order_polygon(const std::vector<cv::Point2f>& points);

// 產生 16 種 (8 個起點 x 正/反方向) cyclic 排列
std::vector<std::vector<cv::Point2f>> cyclic_variants(const std::vector<cv::Point2f>& points);

// 通用相似度變換估計 (任意 src -> dst)。內部會即時計算 src 統計量，
// 效能較 estimate_similarity_from_template() 低，僅供通用/測試用途。
std::optional<SimilarityResult> estimate_similarity(const std::vector<cv::Point2f>& src,
                                                      const std::vector<cv::Point2f>& dst);

// 高效版本：src 恆為 config::kCTemplate，統計量已預先快取。
// match_c_shape() 內部的熱路徑呼叫此函式。
std::optional<SimilarityResult> estimate_similarity_from_template(
    const std::vector<cv::Point2f>& dst);

// 對所有候選做形狀比對，回傳誤差最小且通過門檻的最佳匹配
CMatch match_c_shape(const std::vector<BlueCandidate>& candidates);

// 用最佳 C 匹配的 transform，反推綠色 anchor 應在的位置，
// 並與實際偵測到的綠色中心比較
GreenAnchorValidation validate_green_anchor(const CMatch& match,
                                             const std::optional<cv::Point2f>& green_center);

} // namespace tel::core