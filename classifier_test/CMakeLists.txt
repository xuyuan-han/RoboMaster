# cmake needs this line
cmake_minimum_required(VERSION 2.8)

# Define project name
project(OpenCV_Armor_Finder_project)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "-std=c++11")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
#set(CMAKE_BUILD_TYPE Release) 这句有的话需要注释掉

# set(CMAKE_BUILD_TYPE "Debug")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
# set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_FLAGS_DEBUG "-O0 -Wall -g -ggdb")

# SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
# SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
# SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g -ggdb")
# SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3")

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Declare the executable target built from your sources
add_executable(main main.cpp)

# Link your application with OpenCV libraries
target_link_libraries(main LINK_PRIVATE ${OpenCV_LIBS})
