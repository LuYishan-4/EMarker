// ============================================================
// pnp_solver.hpp
//
// 職責：
//   1. C_TEMPLATE -> object points (mm)
//   2. SQPnP 求解 rvec/tvec
//   3. Reprojection error 計算
//   4. rvec -> Roll/Pitch/Yaw (deg)
//
// 設計重點：
//   - object points 依賴 config::kCTemplate + kModelMmPerPixel，
//     兩者皆為 compile-time 常數，因此以 static lazy-init 快取，
//     避免每個 frame 重新配置/轉換 8 個點 (對應 Python 版
//     get_c_object_points() 每次呼叫都重建 array 的開銷)。
//   - CameraIntrinsics 由 calibration 模組注入 (依賴反轉)，
//     pnp_solver 本身不知道 yml/npz 檔案存在，只吃 cv::Mat。
//   - solve_pnp_and_evaluate() 是給 pipeline 呼叫的單一入口，
//     內部整合 solve + reprojection + 門檻判定，回傳的
//     PoseResult.valid 已包含 kMaxReprojectionError 判斷，
//     使 pipeline 端不需要重複這段邏輯。
// ============================================================
#pragma once
#include <opencv2/core.hpp>
#include <array>
#include <vector>
#include "core/types.hpp"

namespace tel::core {

// 相機內參與畸變係數，由 calibration 模組載入後傳入
struct CameraIntrinsics {
    cv::Matx33d camera_matrix = cv::Matx33d::eye();
    cv::Mat dist_coeffs; // Nx1 or 1xN, CV_64F

    bool valid() const { return !dist_coeffs.empty(); }
};

// C_TEMPLATE 轉換為 3D object points (Z=0 平面, 單位 mm)，
// 結果為 static 快取，多次呼叫僅第一次計算。
const std::vector<cv::Point3f>& get_c_object_points();

// 純 PnP 求解 (SQPnP)，不含門檻判定。
// image_points 需恰為 8 點，且順序須與 get_c_object_points() 對齊
// (亦即與 config::kCTemplate 逐點對應)。
// 失敗 (求解失敗 / 點數不符) 回傳 PoseResult{valid=false}。
PoseResult solve_pnp(const std::vector<cv::Point2f>& image_points,
                      const CameraIntrinsics& intrinsics);

// 計算 reprojection error (所有點誤差之平均 L2 距離, px)
double calculate_reprojection_error(const std::vector<cv::Point2f>& image_points,
                                     const cv::Mat& rvec, const cv::Mat& tvec,
                                     const CameraIntrinsics& intrinsics);

// 整合入口：求解 PnP -> 計算 reprojection -> 依
// config::kMaxReprojectionError 判定 valid。
// 這是 pipeline 唯一需要呼叫的函式。
PoseResult solve_pnp_and_evaluate(const std::vector<cv::Point2f>& image_points,
                                   const CameraIntrinsics& intrinsics);

// rvec (Rodrigues) -> Roll/Pitch/Yaw，單位 degree。
// 沿用 Python 版的 R[2,1]/R[2,2] ... 慣例 (XYZ 外旋順序)，
// 並保留 gimbal-lock (sy < 1e-6) 分支以維持數值穩定性。
cv::Vec3d rotation_to_rpy_deg(const cv::Mat& rvec);

} // namespace tel::core