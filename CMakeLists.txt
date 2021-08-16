cmake_minimum_required(VERSION 3.6)

cmake_policy(SET CMP0072 NEW)   # opengl policy (to get rid of warning)
set(OpenGL_GL_PREFERENCE GLVND) #

project(graphene VERSION 0.0.2) # project name and version
configure_file(inc/version/version.hpp.in ../inc/version/version.hpp)

set(TARGET graphene)
set(GRPH_ENABLE_CUDA TRUE)                         # set to TRUE to enable CUDA
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.2/") # set to location of CUDA source
set(GRPH_CUDA_ARCH        "sm_75")                 # set for specific graphics card

set(CMAKE_CXX_FLAGS_DEBUG_INIT   "-O0 -g -Wall -pg")             # debug
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O2 -Wall -ftree-vectorize") # release

# #### build internal libs ####
include_directories(
  inc/
  inc/version
  )

# # build CUDA components
if(GRPH_ENABLE_CUDA) # enable CUDA
  set(CUDACXX "nvcc")
  set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
  set(CMAKE_CUDA_FLAGS "-g --expt-relaxed-constexpr") # --extended-lambda") # --device-c")
  add_definitions(-DENABLE_CUDA)
  enable_language(CUDA)
  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)         # CUDA 11
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif()
  
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
  set_target_properties(grphcuda PROPERTIES CXX_STANDARD 17)
  set(CUDA_LIBRARIES grphcuda)
endif()

#### build external libs ####
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_GLOBAL}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS_GLOBAL}")
set(CMAKE_LIBRARY_LINKER_FLAGS "${CMAKE_LIBRARY_LINKER_FLAG_GLOBAL}")
include_directories(
  libs/imgui
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
  src/main.cpp
  src/settingForm.cpp
  src/simWindow.cpp
  src/tabMenu.cpp
  )
set_target_properties(${TARGET} PROPERTIES CXX_STANDARD 17)
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
