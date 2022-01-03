cmake_minimum_required(VERSION 3.16)

set(TARGET graphene)
project(${TARGET} VERSION 0.1.2)
configure_file(inc/version/version.hpp.in ../inc/version/version.hpp)

# CUDA config
set(CUDA_TOOLKIT_VERSION  "11.4")  # CUDA version
set(GRPH_CUDA_ARCH        "sm_75") # GPU architecture (https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)
set(GRPH_CUDA_CODE        "compute_86") # GPU code (?)
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-${CUDA_TOOLKIT_VERSION}" ) # CUDA source root

# create verbose Makefile for debug build
if(("${CMAKE_BUILD_TYPE}" MATCHES "DEBUG") OR ("${CMAKE_BUILD_TYPE}" MATCHES "RELWITHDEBINFO")) # verbose builds
  set(CMAKE_VERBOSE_MAKEFILE 1)
endif()


include_directories(
  inc/
  inc/ui/
  inc/version
  )

#### CUDA components ####
set(CUDACXX "nvcc")
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
enable_language(CUDA)

include_directories(
  inc/cuda
  inc/cuda/helper
  ${CUDA_TOOLKIT_ROOT_DIR}/include
  )
add_library(grphcuda STATIC
  src/cuda/draw.cu
  src/cuda/field.cu
  src/cuda/fluid.cu
  src/cuda/em.cu
  src/cuda/render.cu
  src/cuda/vlines.cu
  )

set_target_properties(grphcuda PROPERTIES CXX_STANDARD 20)
set(CUDA_LIBRARIES grphcuda)

set(GRPH_CUDA_FLAGS_GLOBAL --expt-relaxed-constexpr -use_fast_math --std=c++17 # flags for all CUDA builds
  -arch ${GRPH_CUDA_ARCH} # set architecture and compute code
  # -gencode=arch=compute_52,code=sm_52
  # -gencode=arch=compute_60,code=sm_60
  # -gencode=arch=compute_61,code=sm_61
  # -gencode=arch=compute_70,code=sm_70
  -gencode=arch=compute_75,code=sm_75
  #-gencode=arch=compute_80,code=sm_80
  # -gencode=arch=compute_86,code=sm_86
  # -gencode=arch=compute_86,code=compute_86
  )

if     ("${CMAKE_BUILD_TYPE}" MATCHES "DEBUG")          # DEBUG
  target_compile_options(grphcuda PRIVATE ${GRPH_CUDA_FLAGS_GLOBAL} -O0 -pg -g)
  target_link_options   (grphcuda PRIVATE ${GRPH_CUDA_FLAGS_GLOBAL} -O0 -pg -g)
elseif ("${CMAKE_BUILD_TYPE}" MATCHES "RELWITHDEBINFO") # TEST
  target_compile_options(grphcuda PRIVATE ${GRPH_CUDA_FLAGS_GLOBAL} -O2 -pg -Xcompiler="-ftree-vectorize" -Xcompiler="-mavx")
  target_link_options   (grphcuda PRIVATE ${GRPH_CUDA_FLAGS_GLOBAL} -O2 -pg)
elseif ("${CMAKE_BUILD_TYPE}" MATCHES "RELEASE")        # RELEASE
  target_compile_options(grphcuda PRIVATE ${GRPH_CUDA_FLAGS_GLOBAL} -O3 -Xcompiler="-ftree-vectorize" -Xcompiler="-mavx" -Xptxas="-suppress-stack-size-warning")
  target_link_options   (grphcuda PRIVATE ${GRPH_CUDA_FLAGS_GLOBAL} -O3)
endif()

#### dependencies (built from source) ####
include_directories(
  libs
  libs/imgui                # Dear ImGui,
  libs/imgui/backends
  libs/imgui/misc/freetype
  libs/stb                  # stb_image (header only)
  libs/json/include         # json      (header only)
  )
add_subdirectory(libs)
set(EXTRA_LIBS imgui)
set(GLFW_LIBRARIES glfw) # GLFW (submodule)

#### find external dependencies ####
find_package(OpenGL REQUIRED)              # opengl
include_directories(${OPENGL_INCLUDE_DIRS})
link_libraries(${OPENGL_LIBRARY_DIRS})
find_package(GLEW REQUIRED)                # GLEW
set(GLEW_LIBRARIES GLEW::GLEW)

# OS-dependent
if     (UNIX)  # Linux (GTK)
  set(LINUX_USE_GTK 1) # set to false to link without GTK (via zenity)
  message("==> Compiling for Linux")
elseif (WIN32) # Windows -- WIP
  message("==> Compiling for Windows")
  message("TODO: Windows support!")
elseif (APPLE) # macOS -- WIP
  message("==> Compiling for macOS")
  message("TODO: macOS support!")
endif ()

# print found libraries
message(STATUS " =")
message(STATUS " ==")
message(STATUS " ===")
message(STATUS " === CUDA  LIBS: ==> ${CUDA_LIBRARIES}")
message(STATUS " === GLEW  LIBS: ==> ${GLEW_LIBRARIES}")
message(STATUS " === GLFW  LIBS: ==> ${GLFW_LIBRARIES}")
message(STATUS " === GL    LIBS: ==> ${OPENGL_LIBRARIES}")
message(STATUS " === EXTRA LIBS: ==> ${EXTRA_LIBS}")
message(STATUS " ===")
message(STATUS " ==")
message(STATUS " =")

# link executable
add_executable(${TARGET}
  src/main.cpp
  src/cuda-tools.cpp
  src/image.cpp
  src/glShader.cpp
  src/keyFrameWidget.cpp
  src/keyBinding.cpp
  src/keyManager.cpp
  src/output.cpp
  src/frameWriter.cpp
  src/simWindow.cpp
  src/ui/settingForm.cpp
  src/ui/tabMenu.cpp
  src/ui/toolbar.cpp
  )
set_target_properties (${TARGET} PROPERTIES CXX_STANDARD 20)
target_compile_definitions(${TARGET} PRIVATE GLEW_STATIC JSON_HAS_CPP_17)

set(CMAKE_CXX_FLAGS_GLOBAL "") # flags for all C++ builds

if     ("${CMAKE_BUILD_TYPE}" MATCHES "DEBUG")  # debug flags
  target_compile_options(${TARGET} PRIVATE ${CMAKE_CXX_FLAGS_GLOBAL} -O0 -g -pg)
  target_link_options   (${TARGET} PRIVATE ${CMAKE_CXX_FLAGS_GLOBAL} -O0 -g -pg)
elseif ("${CMAKE_BUILD_TYPE}" MATCHES "RELWITHDEBINFO") # release flags
  target_compile_options(${TARGET} PRIVATE ${CMAKE_CXX_FLAGS_GLOBAL} -O2 -pg)
  target_link_options   (${TARGET} PRIVATE ${CMAKE_CXX_FLAGS_GLOBAL} -O2 -pg)
elseif ("${CMAKE_BUILD_TYPE}" MATCHES "RELEASE") # release flags
  target_compile_options(${TARGET} PRIVATE ${CMAKE_CXX_FLAGS_GLOBAL} -O3 -ftree-vectorize -mavx -flto)
  target_link_options   (${TARGET} PRIVATE ${CMAKE_CXX_FLAGS_GLOBAL} -O3 -ftree-vectorize -mavx -flto)
endif ()

target_link_libraries(${TARGET}
  ${CUDA_LIBRARIES}
  ${GLEW_LIBRARIES}
  ${GLFW_LIBRARIES}
  ${OPENGL_LIBRARIES}
  ${EXTRA_LIBS}
  )
