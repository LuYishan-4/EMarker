#include "core/color_mask.hpp"
#include <opencv2/imgproc.hpp>

namespace tel::core {

cv::Mat to_hsv(const cv::Mat& bgr_frame) {
    cv::Mat hsv;
    cv::cvtColor(bgr_frame, hsv, cv::COLOR_BGR2HSV);
    return hsv;
}

cv::Mat make_mask(const cv::Mat& hsv, const config::HsvRange& r) {
    const cv::Scalar low(std::max(0, r.h - r.h_tol), std::max(0, r.s - r.s_tol),
                          std::max(0, r.v - r.v_tol));
    const cv::Scalar high(std::min(179, r.h + r.h_tol), std::min(255, r.s + r.s_tol),
                           std::min(255, r.v + r.v_tol));

    cv::Mat mask;
    cv::inRange(hsv, low, high, mask);

    // kernel 建立成本極低，但仍以 static 避免重複配置
    static const cv::Mat kernel =
        cv::getStructuringElement(cv::MORPH_RECT, {config::kMorphKernel, config::kMorphKernel});

    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    return mask;
}

} // namespace tel::core