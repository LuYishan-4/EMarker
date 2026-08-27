// ============================================================
// types.hpp
//
// Pipeline 各階段之間傳遞的純資料結構。
// ============================================================
#pragma once

#include <array>
#include <vector>

#include <opencv2/core.hpp>

namespace tel::core {

// ============================================================
// BlueCandidate
//
// 單一 C 候選。
// ============================================================

struct BlueCandidate {

    // --------------------------------------------------------
    // C 的 8 個影像座標
    //
    // 順序必須與 config::kCTemplate 對應。
    // --------------------------------------------------------

    std::array<cv::Point2f, 8> points{};


    // --------------------------------------------------------
    // 原始 C contour
    //
    // 主要供 debug / visualization 使用。
    // --------------------------------------------------------

    std::vector<cv::Point> contour;


    // --------------------------------------------------------
    // C 面積
    // --------------------------------------------------------

    double area = 0.0;


    // ========================================================
    // Homography
    // ========================================================

    // Template -> image
    cv::Mat homography;

    // Homography 根據 C 模型推算出的 Green center
    cv::Point2f expected_green_h{0.f, 0.f};

    // 8 個 C 點經 Homography reproject 後的 RMS
    double homography_reprojection_rms = 0.0;

    // Homography 驗證是否通過
    bool homography_valid = false;


    // ========================================================
    // SQPnP
    // ========================================================

    // Rotation vector
    cv::Vec3d rvec{0.0, 0.0, 0.0};

    // Translation vector
    cv::Vec3d tvec{0.0, 0.0, 0.0};

    // SQPnP 根據 C 平面模型推算出的 Green center
    cv::Point2f expected_green_pnp{0.f, 0.f};

    // 8 個 C 點經 PnP reproject 後的 RMS
    double pnp_reprojection_rms = 0.0;

    // SQPnP 驗證是否通過
    bool pnp_valid = false;


    // ========================================================
    // Homography / PnP cross validation
    // ========================================================

    // 兩種模型預測的 Green center 差距
    double green_prediction_error = 0.0;


    // ========================================================
    // Final validation
    // ========================================================

    // C 是否通過所有必要幾何驗證
    bool valid = false;
};


// ============================================================
// Similarity transform
// ============================================================

struct SimilarityResult {

    double scale = 0.0;

    double rotation_deg = 0.0;

    cv::Point2f translation{
        0.f,
        0.f
    };

    std::array<cv::Point2f, 8> predicted{};

    double rms_error = 0.0;
};


// ============================================================
// CMatch
//
// 如果 pipeline 最後只需要一個最佳 C，使用這個。
// ============================================================

struct CMatch {

    bool found = false;

    // 排序後、與 template 對齊的 8 點
    std::array<cv::Point2f, 8> points{};

    // Template transform 後的預測點
    std::array<cv::Point2f, 8> predicted{};

    double shape_error = 0.0;

    double scale = 0.0;

    double rotation_deg = 0.0;

    cv::Point2f translation{
        0.f,
        0.f
    };

    std::vector<cv::Point> contour;

    double area = 0.0;
};


// ============================================================
// Green anchor validation
// ============================================================

struct GreenAnchorValidation {

    // 是否有實際偵測到 Green
    bool has_reference = false;

    // C 根據幾何模型預測的 Green center
    cv::Point2f predicted{
        0.f,
        0.f
    };

    // 預測中心與實際 Green center 的距離
    double error = 0.0;

    // Green 驗證是否通過
    bool valid = false;
};


// ============================================================
// PnP result
//
// 如果 pipeline 需要保留通用 PoseResult。
// ============================================================

struct PoseResult {

    bool valid = false;

    cv::Mat rvec;

    cv::Mat tvec;

    double reprojection_error = 0.0;
};


// ============================================================
// FrameResult
//
// 一個 frame 的完整結果。
// ============================================================

struct FrameResult {

    // --------------------------------------------------------
    // Debug masks
    // --------------------------------------------------------

    cv::Mat green_mask;

    cv::Mat blue_mask;


    // --------------------------------------------------------
    // Green
    // --------------------------------------------------------

    bool green_found = false;

    cv::Point2f green_center{
        0.f,
        0.f
    };


    // --------------------------------------------------------
    // 所有 C candidates
    //
    // 這裡就是你現在要求的：
    //
    //     blue_candidates
    //
    // 每個元素都是一個獨立 C。
    // --------------------------------------------------------

    std::vector<BlueCandidate> candidates;


    // --------------------------------------------------------
    // 最終選中的 C
    // --------------------------------------------------------

    CMatch c_match;


    // --------------------------------------------------------
    // Green anchor 驗證
    // --------------------------------------------------------

    GreenAnchorValidation anchor;


    // --------------------------------------------------------
    // 最終 Pose
    // --------------------------------------------------------

    PoseResult pose;


    // --------------------------------------------------------
    // 最終結果
    // --------------------------------------------------------

    bool detected() const {
        return pose.valid;
    }
};

} // namespace tel::core