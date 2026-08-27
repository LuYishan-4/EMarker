#include "calibration/camera_calibration.hpp"
#include "core/color_mask.hpp"
#include "core/config.hpp"
#include "core/green_anchor.hpp"
#include "pipeline/emarker.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

// ============================================================
// Draw text
// ============================================================

namespace {

void draw_text(cv::Mat &frame, const std::string &text, int row,
               const cv::Scalar &color = cv::Scalar(255, 255, 255)) {

  cv::putText(frame, text, cv::Point(15, 30 + row * 27),
              cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv::LINE_AA);
}

// ============================================================
// Draw C 8 points
// ============================================================

void draw_c_points(cv::Mat &frame, const std::array<cv::Point2f, 8> &points) {

  for (int i = 0; i < 8; ++i) {

    cv::circle(frame, points[i], 4, cv::Scalar(255, 0, 0), -1, cv::LINE_AA);

    cv::putText(frame, "C" + std::to_string(i),
                points[i] + cv::Point2f(6.0f, -6.0f), cv::FONT_HERSHEY_SIMPLEX,
                0.45, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
  }

  for (int i = 0; i < 8; ++i) {

    const int next = (i + 1) % 8;

    cv::line(frame, points[i], points[next], cv::Scalar(255, 0, 0), 2,
             cv::LINE_AA);
  }
}

// ============================================================
// Draw Green center
// ============================================================

void draw_green_center(cv::Mat &frame, const cv::Point2f &point,
                       const cv::Scalar &color, const std::string &text) {

  cv::circle(frame, point, 7, color, 2, cv::LINE_AA);

  cv::line(frame, point - cv::Point2f(12.0f, 0.0f),
           point + cv::Point2f(12.0f, 0.0f), color, 2, cv::LINE_AA);

  cv::line(frame, point - cv::Point2f(0.0f, 12.0f),
           point + cv::Point2f(0.0f, 12.0f), color, 2, cv::LINE_AA);

  cv::putText(frame, text, point + cv::Point2f(10.0f, 20.0f),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
}

} // namespace

// ============================================================
// MAIN
// ============================================================

int main(int argc, char **argv) {

  // ========================================================
  // Arguments
  //
  // argv[1] = video
  // argv[2] = calibration
  // ========================================================

  const std::string video_path = argc > 1 ? argv[1] : "video.mp4";

  std::string calibration_path;

  bool no_display = false;

  for (int i = 2; i < argc; ++i) {

    const std::string arg = argv[i];

    if (arg == "--no-display") {

      no_display = true;

    } else if (calibration_path.empty()) {

      calibration_path = arg;
    }
  }

  // ========================================================
  // Open video
  // ========================================================

  cv::VideoCapture video(video_path);

  if (!video.isOpened()) {

    std::cerr << "Cannot open video: " << video_path << '\n';

    return 1;
  }

  const int total_frames =
      static_cast<int>(video.get(cv::CAP_PROP_FRAME_COUNT));

  std::cout << "Video frames: " << total_frames << '\n';

  // ========================================================
  // Load calibration
  // ========================================================

  tel::calibration::CameraCalibration calibration;

  if (!calibration_path.empty()) {

    calibration = tel::calibration::load_calibration(calibration_path);

  } else {

    calibration = tel::calibration::load_calibration("");
  }

  // ========================================================
  // Camera matrix
  // ========================================================

  cv::Matx33d camera_matrix = tel::config::kFallbackCameraMatrix;

  if (calibration.camera_matrix.rows == 3 &&
      calibration.camera_matrix.cols == 3) {

    for (int row = 0; row < 3; ++row) {

      for (int col = 0; col < 3; ++col) {

        camera_matrix(row, col) =
            calibration.camera_matrix.at<double>(row, col);
      }
    }
  }

  cv::Mat dist_coeffs = calibration.dist_coeffs.clone();

  // ========================================================
  // Pipeline
  // ========================================================

  tel::pipeline::CMarkerPipeline pipeline(camera_matrix, dist_coeffs);

  // ========================================================
  // Current frame
  // ========================================================

  int current_frame = 0;

  // ========================================================
  // Statistics
  // ========================================================

  int processed_frames = 0;
  int frame_with_green = 0;
  int frame_with_c = 0;
  int total_valid_markers = 0;

  double best_green_error = std::numeric_limits<double>::max();

  double best_pnp_error = std::numeric_limits<double>::max();

  double best_homography_error = std::numeric_limits<double>::max();

  // ========================================================
  // Frame processing lambda
  // ========================================================

  auto process_frame = [&](int frame_index) -> bool {
    if (frame_index < 0 || frame_index >= total_frames) {

      return false;
    }

    // ----------------------------------------------------
    // Seek
    // ----------------------------------------------------

    video.set(cv::CAP_PROP_POS_FRAMES, frame_index);

    cv::Mat frame;

    if (!video.read(frame)) {

      std::cerr << "Cannot read frame " << frame_index << '\n';

      return false;
    }

    // ====================================================
    // HSV
    // ====================================================

    const cv::Mat hsv = tel::core::to_hsv(frame);

    // ====================================================
    // GREEN BINARY
    // ====================================================

    const cv::Mat green_mask =
        tel::core::make_mask(hsv, tel::config::kGreenHsv);

    // ====================================================
    // BLUE BINARY
    // ====================================================

    const cv::Mat blue_mask = tel::core::make_mask(hsv, tel::config::kBlueHsv);

    // ====================================================
    // Pipeline
    // ====================================================

    const std::vector<tel::pipeline::CMarker> markers =
        pipeline.process(blue_mask, green_mask);

    // ====================================================
    // Green debug
    // ====================================================

    const auto green_result = tel::core::find_green_anchors(green_mask);

    // ====================================================
    // Statistics
    // ====================================================

    ++processed_frames;

    if (green_result.has_value() && !green_result->centers.empty()) {

      ++frame_with_green;
    }

    if (!markers.empty()) {

      ++frame_with_c;

      total_valid_markers += static_cast<int>(markers.size());
    }

    // ====================================================
    // Original display
    // ====================================================

    cv::Mat display = frame.clone();

    // ----------------------------------------------------
    // Draw all Green
    // ----------------------------------------------------

    if (green_result.has_value()) {

      for (const auto &center : green_result->centers) {

        draw_green_center(display, center, cv::Scalar(0, 255, 0), "Green");
      }
    }

    // ----------------------------------------------------
    // Draw valid C markers
    // ----------------------------------------------------

    for (const auto &marker : markers) {
      std::cout << "marker "
                   "sizelkollllllllllffffffffffffffffffffffffffffffffffffffffff"
                   "fffffffffffflllllllllj: "
                << marker.c_points().size() << std::endl;

      draw_c_points(display, marker.c_points());

      draw_green_center(display, marker.green_center(), cv::Scalar(0, 255, 0),
                        "G actual");

      draw_green_center(display, marker.expected_green(),
                        cv::Scalar(0, 255, 255), "G predicted");

      best_green_error = std::min(best_green_error, marker.green_error());

      best_pnp_error = std::min(best_pnp_error, marker.pnp_reprojection_rms());

      best_homography_error =
          std::min(best_homography_error, marker.homography_reprojection_rms());
    }

    std::cout << "marker "
                 "sizelkollllllllllffffffffffffffffffffffffffffffffffffffffff"
                 "fffffffffffflllllllllj: "
              << markers.size() << std::endl;
    draw_text(display,
              "Frame: " + std::to_string(frame_index) + " / " +
                  std::to_string(total_frames - 1),
              0);

    draw_text(display,
              "Green: " + std::to_string(green_result.has_value()
                                             ? green_result->centers.size()
                                             : 0),
              1);

    draw_text(display, "Valid C: " + std::to_string(markers.size()), 2,
              markers.empty() ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0));

    draw_text(display, "A/D or Arrow: previous/next frame", 3);

    // ====================================================
    // GREEN BINARY DISPLAY
    // ====================================================

    cv::Mat green_display;

    cv::cvtColor(green_mask, green_display, cv::COLOR_GRAY2BGR);

    draw_text(green_display, "GREEN BINARY", 0);

    draw_text(green_display,
              "Non-zero: " + std::to_string(cv::countNonZero(green_mask)), 1);

    // ====================================================
    // BLUE BINARY DISPLAY
    // ====================================================

    cv::Mat blue_display;

    cv::cvtColor(blue_mask, blue_display, cv::COLOR_GRAY2BGR);

    draw_text(blue_display, "BLUE BINARY", 0);

    draw_text(blue_display,
              "Non-zero: " + std::to_string(cv::countNonZero(blue_mask)), 1);

    // ====================================================
    // GREEN + BLUE BINARY
    //
    // Green = Green
    // Blue  = Blue
    // ====================================================

    cv::Mat combined = cv::Mat::zeros(frame.size(), CV_8UC3);

    // Green mask → green channel
    combined.setTo(cv::Scalar(0, 255, 0), green_mask);

    // Blue mask → blue channel
    combined.setTo(cv::Scalar(255, 0, 0), blue_mask);

    // Both → cyan
    cv::Mat both;
    cv::bitwise_and(green_mask, blue_mask, both);

    combined.setTo(cv::Scalar(255, 255, 0), both);

    draw_text(combined, "GREEN + BLUE BINARY", 0);

    draw_text(combined,
              "Green pixels: " + std::to_string(cv::countNonZero(green_mask)),
              1);

    draw_text(combined,
              "Blue pixels: " + std::to_string(cv::countNonZero(blue_mask)), 2);

    // ====================================================
    // Show
    // ====================================================

    if (!no_display) {

      cv::imshow("Original", display);

      cv::imshow("HSV Green Binary", green_display);

      cv::imshow("HSV Blue Binary", blue_display);

      cv::imshow("Green + Blue Binary", combined);
    }

    // ====================================================
    // Console debug
    // ====================================================
    //
    std::cout << "marker "
                 "sizelkollllllllllffffffffffffffffffffffffffffffffffffffffff"
                 "fffffffffffflllllllllj: "
              << markers.size() << std::endl;
    std::cout
        << "\n"
        << "============================================================\n"
        << "FRAME " << frame_index << " / " << total_frames - 1 << "\n"
        << "============================================================\n";

    std::cout << "[GREEN]\n";

    if (green_result.has_value()) {

      std::cout << "  count           : " << green_result->centers.size()
                << '\n';

      for (std::size_t i = 0; i < green_result->centers.size(); ++i) {

        std::cout << "  Green[" << i << "]\n"
                  << "    center        : (" << green_result->centers[i].x
                  << ", " << green_result->centers[i].y << ")\n"
                  << "    area          : " << green_result->areas[i] << '\n'
                  << "    area_ratio    : " << green_result->area_ratios[i]
                  << '\n'
                  << "    geometry      : "
                  << (green_result->area_ratios[i] >= 0.7f ? "PASS" : "FAIL")
                  << '\n';
      }

    } else {

      std::cout << "  count           : 0\n";
    }

    // ====================================================
    // Binary statistics
    // ====================================================

    std::cout << "\n"
              << "[BINARY]\n"
              << "  Green non-zero : " << cv::countNonZero(green_mask) << '\n'
              << "  Blue non-zero  : " << cv::countNonZero(blue_mask) << '\n';

    // ====================================================
    // Pipeline result
    // ====================================================

    std::cout << "\n"
              << "[PIPELINE RESULT]\n"
              << "  valid markers   : " << markers.size() << '\n'
              << "  RESULT          : "
              << (markers.empty() ? "NO VALID MARKER" : "VALID") << '\n';

    std::cout.flush();

    return true;
  };

  // ========================================================
  // Initial frame
  // ========================================================

  process_frame(current_frame);

  // ========================================================
  // Frame stepping
  // ========================================================

  if (no_display) {

    // 無頭模式：自動跑完所有 frame
    while (current_frame < total_frames - 1) {

      ++current_frame;

      process_frame(current_frame);
    }

  } else {

    while (true) {

      const int key = cv::waitKey(0) & 0xff;

      // ----------------------------------------------------
      // ESC / Q
      // ----------------------------------------------------

      if (key == 27 || key == 'q' || key == 'Q') {

        break;
      }

      // ----------------------------------------------------
      // Next frame
      //
      // D / →
      // ----------------------------------------------------

      if (key == 'd' || key == 'D' || key == 83) {

        if (current_frame < total_frames - 1) {

          ++current_frame;

          process_frame(current_frame);
        }

        continue;
      }

      // ----------------------------------------------------
      // Previous frame
      //
      // A / ←
      // ----------------------------------------------------

      if (key == 'a' || key == 'A' || key == 81) {

        if (current_frame > 0) {

          --current_frame;

          process_frame(current_frame);
        }

        continue;
      }
    }
  }

  // ========================================================
  // Cleanup
  // ========================================================

  cv::destroyAllWindows();

  // ========================================================
  // Statistics
  // ========================================================

  std::cout << "\n"
            << "============================================================\n"
            << "TEL Marker Test Result\n"
            << "============================================================\n"
            << "Processed frames       : " << processed_frames << '\n'
            << "Frames with Green      : " << frame_with_green << '\n'
            << "Frames with C          : " << frame_with_c << '\n'
            << "Total valid markers    : " << total_valid_markers << '\n';

  if (best_green_error < std::numeric_limits<double>::max()) {

    std::cout << "Best Green error       : " << best_green_error << " px\n";
  }

  if (best_pnp_error < std::numeric_limits<double>::max()) {

    std::cout << "Best PnP RMS           : " << best_pnp_error << " px\n";
  }

  if (best_homography_error < std::numeric_limits<double>::max()) {

    std::cout << "Best Homography RMS    : " << best_homography_error
              << " px\n";
  }

  std::cout << "============================================================\n";

  return 0;
}
