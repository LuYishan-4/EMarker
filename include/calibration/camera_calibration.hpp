// ============================================================
// camera_calibration.hpp
//
// 職責：讀取由 tools/npz_to_yaml.py 預先轉換好的 yaml/json/xml
// 相機標定檔 (camera_matrix, dist_coeffs)。
//
// C++ 端不解析 .npz，交由 Python 工具離線轉檔，
// 這裡只做 OpenCV FileStorage 讀取，保持職責單純。
// ============================================================
#pragma once

#include <opencv2/core.hpp>
#include <string>

namespace tel::calibration {

struct CameraCalibration {
    cv::Mat camera_matrix; // 3x3, CV_64F
    cv::Mat dist_coeffs;   // Nx1, CV_64F
    bool loaded_from_file = false;
};

// 嘗試從 yaml/json/xml 讀取標定檔；失敗則回傳內建 fallback 參數，
// loaded_from_file 標記讀取是否成功，供上層決定要不要警告使用者。
CameraCalibration load_calibration(const std::string& path);

} // namespace tel::calibration