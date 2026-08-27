// ============================================================
// config.hpp
//
// 所有可調參數集中於此。
// 純數值常數，不含任何邏輯，方便日後做成 yaml 熱載入。
// ============================================================
#pragma once

#include <array>
#include <cstdint>

#include <opencv2/core.hpp>

namespace tel::config {

// ============================================================
// Model scale
// ============================================================

// 模型座標與實際尺寸的比例。
// 目前演算法本身只要求所有模型座標使用相同單位。
// 若之後需要輸出實際距離 / mm，可由此換算。
inline constexpr double kModelMmPerPixel = 400.0/235.0;


// ============================================================
// C MODEL
//
// C 在自身平面座標系中的固定 8 個點。
// Z = 0
//
// 座標原點：Green square 左上角
//
// X → right
// Y → down
//
// 最新模型：
// P0: (-23.500, -25.533)
// P1: (-7.481,  -10.233)
// P2: (24.500,  -25.533)
// P3: (24.000,  -11.521)
// P4: (23.814,   30.967)
// P5: (23.500,   46.467)
// P6: (-7.474,   28.967)
// P7: (-23.500,  46.467)
// ============================================================

inline const std::array<cv::Point2d, 8> kCTemplate = {{
    {-23.500, -25.533},
    {-7.481,  -10.233},
    { 24.500, -25.533},
    { 24.000, -11.521},
    { 23.814,  30.967},
    { 23.500,  46.467},
    {-7.474,  28.967},
    {-23.500,  46.467},
}};


// ============================================================
// Green anchor geometry
//
// Green top-left = model origin
//
// Green center:
//     X = 8.712
//     Y = 9.162
// ============================================================

inline const cv::Point2d kGreenCenterFromOrigin{
    8.712,
    9.162
};


// ============================================================
// HSV color centers / tolerances
// ============================================================

struct HsvRange {
    int h;
    int s;
    int v;

    int h_tol;
    int s_tol;
    int v_tol;
};

/*
BLUE
Center = [91, 200, 255]
Tol = [8, 107, 42]
Range = [83, 93, 213] ~ [99, 255, 255]

GREEN
Center = [75, 135, 255]
Tol = [8, 45, 87]
Range = [67, 90, 168] ~ [83, 180, 255]
*/

inline constexpr HsvRange kGreenHsv{
    /* h     */ 75,
    /* s     */ 135,
    /* v     */ 255,
    /* h_tol */ 8,
    /* s_tol */ 45,
    /* v_tol */ 87
};

inline constexpr HsvRange kBlueHsv{
    /* h     */ 91,
    /* s     */ 200,
    /* v     */ 255,
    /* h_tol */ 8,
    /* s_tol */ 107,
    /* v_tol */ 42
};


// ============================================================
// Detection thresholds
// ============================================================

inline constexpr double kMinGreenArea = 10.0;
inline constexpr double kMinBlueArea  = 20.0;

inline constexpr int kMorphKernel = 3;


// ============================================================
// Shape / contour detection
// ============================================================

// C contour 最少需要多少個原始 contour points。
// 太少通常代表輪廓不可靠。
inline constexpr std::size_t kMinContourPoints = 10;

// C 最終必須 approximated 成 8 個點。
inline constexpr int kExpectedPointCount = 8;


// ============================================================
// Polygon approximation
//
// 舊值：0.025
// 新值：0.010
//
// 注意：這裡直接覆蓋原本 kApproxEpsilonRatio。
// ============================================================

inline constexpr double kApproxEpsilonRatio = 0.010;


// ============================================================
// Homography
// ============================================================

// findHomography() 使用 RANSAC 時的最大像素誤差。
inline constexpr double kHomographyRansacPx = 3.0;

// Homography 對 8 個 C 點重新投影後允許的 RMS。
inline constexpr double kMaxHomographyRms = 3.0;


// ============================================================
// SQPnP
// ============================================================

// SQPnP 對 8 個 C 點重新投影後允許的 RMS。
inline constexpr double kMaxPnPRms = 3.0;


// ============================================================
// Homography / SQPnP cross validation
//
// Homography 預測 Green center
//        VS
// SQPnP 預測 Green center
//
// 兩者距離超過此值，視為幾何模型不一致。
// ============================================================

inline constexpr double kMaxGreenPredictionError = 8.0;


// ============================================================
// General shape / matching thresholds
// ============================================================

// Shape error 最大值。
// 保留給後續 shape matching / CMarker 使用。
inline constexpr double kMaxShapeError = 8.0;

// Green 實際中心與理論中心最大允許距離。
inline constexpr double kMaxGreenAnchorError = 25.0;


// ============================================================
// Scale limits
// ============================================================

// C 投影尺度限制。
// 保留給後續透視 / PnP 合理性檢查。
inline constexpr double kMinScale = 0.15;
inline constexpr double kMaxScale = 20.0;


// ============================================================
// Reprojection
//
// 舊版：
//     kMaxReprojectionError = 10.0
//
// 現在改由：
//     kMaxHomographyRms
//     kMaxPnPRms
//
// 分別控制。
// 不再保留舊的模糊 generic threshold。
// ============================================================


// ============================================================
// Fallback calibration
//
// 若 yaml 讀取失敗時使用。
// ============================================================

inline const cv::Matx33d kFallbackCameraMatrix{
    1112.0566597, 0.0, 954.8446657,
    0.0, 1109.8558990, 489.2472785,
    0.0, 0.0, 1.0
};

inline const std::array<double, 5> kFallbackDist{
    0.0399809499,
    -0.1098228779,
    0.0008024654,
    -0.0012401027,
    0.0505635206
};

} // namespace tel::config