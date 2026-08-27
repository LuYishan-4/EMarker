// ============================================================
// cmaker.hpp
//
// CMarkerPipeline
//
// 一個 CMarker =
//   - C 的 8 個影像座標
//   - C contour / area
//   - 對應的實際 Green center
//   - C 推算出的理論 Green center
//   - Green prediction error
//   - Homography 驗證結果
//   - SQPnP 驗證結果
//   - PnP pose
//
// Pipeline 只輸出已完成配對與驗證的 CMarker。
// ============================================================

#pragma once

#include <array>
#include <opencv2/core.hpp>
#include <vector>

#include "core/types.hpp"

namespace tel::pipeline {

class CMarker {
public:

    CMarker(
        const tel::core::BlueCandidate& candidate,
        const cv::Point2f& green_center,
        const cv::Point2f& expected_green
    )
        : c_points_(candidate.points),
          contour_(candidate.contour),
          area_(candidate.area),

          green_center_(green_center),
          expected_green_(expected_green),

          green_error_(candidate.green_prediction_error),

          homography_(candidate.homography),
          expected_green_h_(candidate.expected_green_h),

          rvec_(candidate.rvec),
          tvec_(candidate.tvec),
          expected_green_pnp_(candidate.expected_green_pnp),

          pnp_reprojection_rms_(
              candidate.pnp_reprojection_rms
          ),

          homography_reprojection_rms_(
              candidate.homography_reprojection_rms
          ),

          homography_valid_(
              candidate.homography_valid
          ),

          pnp_valid_(
              candidate.pnp_valid
          ),

          valid_(
              candidate.valid
          )
    {}

    // ========================================================
    // C 8 points
    // ========================================================

    const std::array<cv::Point2f, 8>& c_points() const {
        return c_points_;
    }

    // ========================================================
    // C contour
    // ========================================================

    const std::vector<cv::Point>& contour() const {
        return contour_;
    }

    // ========================================================
    // C area
    // ========================================================

    double area() const {
        return area_;
    }

    // ========================================================
    // Green
    // ========================================================

    const cv::Point2f& green_center() const {
        return green_center_;
    }

    const cv::Point2f& expected_green() const {
        return expected_green_;
    }

    double green_error() const {
        return green_error_;
    }

    // ========================================================
    // Homography
    // ========================================================

    const cv::Mat& homography() const {
        return homography_;
    }

    const cv::Point2f& expected_green_h() const {
        return expected_green_h_;
    }

    double homography_reprojection_rms() const {
        return homography_reprojection_rms_;
    }

    bool homography_valid() const {
        return homography_valid_;
    }

    // ========================================================
    // SQPnP
    // ========================================================

    const cv::Vec3d& rvec() const {
        return rvec_;
    }

    const cv::Vec3d& tvec() const {
        return tvec_;
    }

    const cv::Point2f& expected_green_pnp() const {
        return expected_green_pnp_;
    }

    double pnp_reprojection_rms() const {
        return pnp_reprojection_rms_;
    }

    bool pnp_valid() const {
        return pnp_valid_;
    }

    // ========================================================
    // Final validation
    // ========================================================

    bool valid() const {
        return valid_;
    }

private:

    // ========================================================
    // C
    // ========================================================

    std::array<cv::Point2f, 8> c_points_{};

    std::vector<cv::Point> contour_;

    double area_ = 0.0;

    // ========================================================
    // Green
    // ========================================================

    cv::Point2f green_center_{};

    cv::Point2f expected_green_{};

    double green_error_ = 0.0;

    // ========================================================
    // Homography
    // ========================================================

    cv::Mat homography_;

    cv::Point2f expected_green_h_{};

    double homography_reprojection_rms_ = 0.0;

    bool homography_valid_ = false;

    // ========================================================
    // SQPnP
    // ========================================================

    cv::Vec3d rvec_{};

    cv::Vec3d tvec_{};

    cv::Point2f expected_green_pnp_{};

    double pnp_reprojection_rms_ = 0.0;

    bool pnp_valid_ = false;

    // ========================================================
    // Final
    // ========================================================

    bool valid_ = false;
};


class CMarkerPipeline {
public:

    CMarkerPipeline(
        const cv::Matx33d& camera_matrix,
        const cv::Mat& dist_coeffs
    );

    std::vector<CMarker> process(
        const cv::Mat& blue_binary,
        const cv::Mat& green_binary
    );

private:

    cv::Matx33d camera_matrix_;

    cv::Mat dist_coeffs_;
};

} // namespace tel::pipeline