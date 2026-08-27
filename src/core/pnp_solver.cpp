// ============================================================
// pnp_solver.cpp
// ============================================================
#include "core/pnp_solver.hpp"
#include "core/config.hpp"
#include <opencv2/calib3d.hpp>
#include <cmath>

namespace tel::core {

constexpr bool kDebugPnP = true;

const std::vector<cv::Point3f>& get_c_object_points() {
    static const std::vector<cv::Point3f> points = [] {
        std::vector<cv::Point3f> pts;
        pts.reserve(8);
        for (const auto& p : config::kCTemplate) {
            pts.emplace_back(
                static_cast<float>(p.x * config::kModelMmPerPixel),
                static_cast<float>(p.y * config::kModelMmPerPixel),
                0.0f);
        }
        return pts;
    }();
    return points;
}

PoseResult solve_pnp(const std::vector<cv::Point2f>& image_points,
                      const CameraIntrinsics& intrinsics) {
    PoseResult result;

    if (image_points.size() != 8) return result;
    if (!intrinsics.valid()) return result;

    const auto& object_points = get_c_object_points();

    cv::Mat rvec, tvec;
    bool ok = false;
    try {
        ok = cv::solvePnP(object_points, image_points,
                           cv::Mat(intrinsics.camera_matrix), intrinsics.dist_coeffs,
                           rvec, tvec, false, cv::SOLVEPNP_SQPNP);
    } catch (const cv::Exception& e) {

        if (kDebugPnP) {

            std::cout
                << "      [PnP] solvePnP exception:\n"
                << "             "
                << e.what()
                << '\n';
        }

        return result;
    }

    if (!ok) return result;

    result.rvec = rvec;
    result.tvec = tvec;
    result.valid = true; // 幾何求解成功；最終有效性仍需 reprojection 檢查
    return result;
}

double calculate_reprojection_error(const std::vector<cv::Point2f>& image_points,
                                     const cv::Mat& rvec, const cv::Mat& tvec,
                                     const CameraIntrinsics& intrinsics) {
    const auto& object_points = get_c_object_points();

    std::vector<cv::Point2f> projected;
    cv::projectPoints(object_points, rvec, tvec,
                       cv::Mat(intrinsics.camera_matrix), intrinsics.dist_coeffs,
                       projected);

    double sum_err = 0.0;
    const size_t n = image_points.size();
    for (size_t i = 0; i < n; ++i) {
        sum_err += cv::norm(projected[i] - image_points[i]);
    }
    return n > 0 ? sum_err / static_cast<double>(n) : 0.0;
}

PoseResult solve_pnp_and_evaluate(const std::vector<cv::Point2f>& image_points,
                                   const CameraIntrinsics& intrinsics) {
    PoseResult result = solve_pnp(image_points, intrinsics);
    if (!result.valid) return result;

    result.reprojection_error =
        calculate_reprojection_error(image_points, result.rvec, result.tvec, intrinsics);

    // 最終有效性 = 幾何求解成功 且 reprojection 誤差在容許範圍內
    result.valid = result.reprojection_error < config::kMaxPnPRms;
    return result;
}

cv::Vec3d rotation_to_rpy_deg(const cv::Mat& rvec) {
    cv::Mat R;
    cv::Rodrigues(rvec, R);

    const double r00 = R.at<double>(0, 0), r10 = R.at<double>(1, 0);
    const double r20 = R.at<double>(2, 0), r21 = R.at<double>(2, 1);
    const double r22 = R.at<double>(2, 2);
    const double r12 = R.at<double>(1, 2), r11 = R.at<double>(1, 1);

    const double sy = std::sqrt(r00 * r00 + r10 * r10);

    double roll, pitch, yaw;
    if (sy >= 1e-6) {
        roll = std::atan2(r21, r22);
        pitch = std::atan2(-r20, sy);
        yaw = std::atan2(r10, r00);
    } else {
        // gimbal lock: 與 Python 版一致的退化分支
        roll = std::atan2(-r12, r11);
        pitch = std::atan2(-r20, sy);
        yaw = 0.0;
    }

    constexpr double kRad2Deg = 180.0 / CV_PI;
    return {roll * kRad2Deg, pitch * kRad2Deg, yaw * kRad2Deg};
}

} // namespace tel::core