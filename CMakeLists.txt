cmake_minimum_required(VERSION 3.12)
project(navigation)

find_package(PCL REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(Armadillo REQUIRED)
# find_package(mlpack REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
/usr/local/include/vtk-8.2
${OPENCV_INCLUDE_DIRS}
${PCL_INCLUDE_DIRS}
${ARMADILLO_INCLUDE_DIRS}
# ${MLPACK_INCLUDE_DIRS}
${EIGEN3_INCLUDE_DIRS}
)

link_directories(
${PCL_LIBRARY_DIRS}
${OpenCV_LIBS}
)

include_directories(include)
file(GLOB_RECURSE SRC ./src/*.cpp)

add_executable(main ${SRC})

target_link_libraries (main
${PCL_LIBRARIES}
${OpenCV_LIBS}
${ARMADILLO_LIBRARIES}
# ${MLPACK_LIBRARIES}
)

install(TARGETS main RUNTIME DESTINATION bin)
