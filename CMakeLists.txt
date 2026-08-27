cmake_minimum_required(VERSION 3.20)

project(EMarker LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


find_package(OpenCV REQUIRED)
if(OpenCV_VERSION VERSION_GREATER_EQUAL "5.0")
    set(EMARKER_OPENCV_CALIB_COMPONENT calib)
else()
    set(EMARKER_OPENCV_CALIB_COMPONENT calib3d)
endif()
find_package(OpenCV REQUIRED COMPONENTS core imgproc ${EMARKER_OPENCV_CALIB_COMPONENT} videoio highgui)

add_library(EMarker
    src/calibration/camera_calibration.cpp
    src/core/blue_candidates.cpp
    src/core/color_mask.cpp
    src/core/green_anchor.cpp
    src/core/pnp_solver.cpp
    src/core/shape_match.cpp
    src/pipeline/emarker.cpp
)

target_include_directories(EMarker PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/calibration
    ${CMAKE_CURRENT_SOURCE_DIR}/include/core
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pipeline
)

target_link_libraries(EMarker PUBLIC ${OpenCV_LIBS})

add_executable(EMarker_video_test main.cpp)
target_link_libraries(EMarker_video_test PRIVATE EMarker ${OpenCV_LIBS})
