#include "calibration/camera_calibration.hpp"
#include "core/config.hpp"

namespace tel::calibration {

CameraCalibration load_calibration(const std::string& path) {
    CameraCalibration calibration;

    if (!path.empty()) {
        try {
            cv::FileStorage storage(path, cv::FileStorage::READ);
            if (storage.isOpened()) {
                storage["camera_matrix"] >> calibration.camera_matrix;
                storage["dist_coeffs"] >> calibration.dist_coeffs;
                calibration.loaded_from_file = !calibration.camera_matrix.empty() &&
                                               !calibration.dist_coeffs.empty() &&
                                               calibration.camera_matrix.rows == 3 &&
                                               calibration.camera_matrix.cols == 3;
            }
        } catch (const cv::Exception&) {
            calibration.loaded_from_file = false;
        }
    }

    if (!calibration.loaded_from_file) {
        calibration.camera_matrix = cv::Mat(3, 3, CV_64F);
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                calibration.camera_matrix.at<double>(row, col) =
                    config::kFallbackCameraMatrix(row, col);
            }
        }
        calibration.dist_coeffs = cv::Mat(static_cast<int>(config::kFallbackDist.size()), 1, CV_64F);
        for (size_t index = 0; index < config::kFallbackDist.size(); ++index) {
            calibration.dist_coeffs.at<double>(static_cast<int>(index), 0) =
                config::kFallbackDist[index];
        }
    }

    return calibration;
}

} // namespace tel::calibration