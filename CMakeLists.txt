cmake_minimum_required(VERSION 3.16)

project(graphene VERSION 0.0.4) # project name and version
configure_file(inc/version/version.hpp.in ../inc/version/version.hpp)

set(TARGET graphene)
set(GRPH_ENABLE_CUDA TRUE)                         # set to TRUE to enable CUDA
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.2/") # set to location of CUDA source
set(GRPH_CUDA_ARCH        "sm_75")                 # set for specific graphics card

# # verbosity
# if("${CMAKE_BUILD_TYPE}" MATCHES "DEBUG") # verbose build for debug
#   set(CMAKE_VERBOSE_MAKEFILE 1)
# endif()

# c++ compiler flags
set(CMAKE_CXX_FLAGS_GLOBAL       "-std=c++20") # NOTE: passing C++ standard manually with -std
set(CMAKE_CXX_FLAGS_DEBUG_INIT   "-O0 -Wall -g -pg")           # debug
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O3 -Wall -ftree-vectorize") # release

# #### build internal libs ####
include_directories(
  inc/
  inc/version
  )

# # build CUDA components
if(GRPH_ENABLE_CUDA) # enable CUDA
  set(CUDACXX "nvcc")
  set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
  set(CMAKE_CUDA_FLAGS         "--expt-relaxed-constexpr -std=c++17") # NOTE: passing C++ standard manually with -std
  set(CMAKE_CUDA_FLAGS_DEBUG   "-g")
  set(CMAKE_CUDA_FLAGS_RELEASE "-Xptxas='-suppress-stack-size-warning'") # suppress warning (due to recursive kernel)
  add_definitions(-DENABLE_CUDA)
  enable_language(CUDA)
  
  # build cuda lib
  include_directories(
    inc/cuda
    inc/cuda/helper
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    )
  add_library(grphcuda STATIC
    src/cuda/draw.cu
    src/cuda/field.cu
    src/cuda/render.cu
    )
  set(CUDA_LIBRARIES grphcuda)
endif()

#### build external libs ####
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_GLOBAL}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS_GLOBAL}")
set(CMAKE_LIBRARY_LINKER_FLAGS "${CMAKE_LIBRARY_LINKER_FLAG_GLOBAL}")
include_directories(
  libs/imgui
  libs/imgui/backends
  libs/imgui/misc/freetype
  libs/stb
  )
add_subdirectory(libs)

# opengl
find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS})
link_libraries(${OPENGL_LIBRARY_DIRS})
# glew
set(GLEW_LIBRARIES GLEW)
# glfw
set(GLFW_LIBRARIES glfw3)
# json
find_package(nlohmann_json 3.2.0 REQUIRED)

# OS-dependent
set(EXTRA_LIBS "")
if (UNIX) # Linux (GTK)
  set(LINUX_USE_GTK 1) # set to false to link without GTK (via zenity)
  message("==> Compiling for Linux...")
elseif (WIN32) # Windows -- WIP
  message("==> Compiling for Windows...")
  message("TODO: Windows support!")
elseif (APPLE) # macOS -- WIP
  message("==> Compiling for macOS...")
  message("TODO: macOS support!")
endif ()

# print found libraries
message(STATUS " =")
message(STATUS " ==")
message(STATUS " ===")
message(STATUS " === CUDA LIBS: ==> ${CUDA_LIBRARIES}")
message(STATUS " === GLEW LIBS: ==> ${GLEW_LIBRARIES}")
message(STATUS " === GLFW LIBS: ==> ${GLFW_LIBRARIES}")
message(STATUS " === GL LIBS:   ==> ${OPENGL_LIBRARIES}")
message(STATUS " ===")
message(STATUS " ==")
message(STATUS " =")

# link executable
message(STATUS "Building graphene...")

add_executable(${TARGET}
  src/cuda-tools.cpp
  src/image.cpp
  src/keyManager.cpp
  src/main.cpp
  src/settingForm.cpp
  src/simWindow.cpp
  src/tabMenu.cpp
  src/units.cpp
  )
# set_target_properties(${TARGET} PROPERTIES CXX_STANDARD 20)
target_compile_options(${TARGET} PRIVATE -Wno-deprecated)

target_link_libraries(${TARGET} PRIVATE
  # grph
  imgui
  glfw
  nlohmann_json::nlohmann_json
  ${OPENGL_LIBRARIES}
  ${GLEW_LIBRARIES}
  ${EXTRA_LIBS}
  ${CUDA_LIBRARIES}
  )
