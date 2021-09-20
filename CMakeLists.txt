cmake_minimum_required(VERSION 3.16)

project(graphene VERSION 0.0.4)
configure_file(inc/version/version.hpp.in ../inc/version/version.hpp)

set(TARGET graphene)
set(GRPH_ENABLE_CUDA TRUE)                        # set to TRUE to enable CUDA
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.2") # set to location of CUDA source
set(GRPH_CUDA_ARCH        "sm_75")                # set for specific graphics card

# set(CMAKE_CXX_STANDARD 20) # NOTE: sets flag -std=gnu++1z --> set manually
set(CMAKE_CXX_FLAGS        -std=c++20)
set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} -flto)

# verbose Makefile for Debug build
if("${CMAKE_BUILD_TYPE}" MATCHES "Debug") # verbose build for debug
  set(CMAKE_VERBOSE_MAKEFILE 1)
endif()


include_directories(
  inc/
  inc/version
  )

#### build CUDA components ####
if(GRPH_ENABLE_CUDA) # enable CUDA
  set(CUDACXX "nvcc")
  set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
  set(CMAKE_CUDA_FLAGS         "-std=c++17") # NOTE: passing C++ standard manually
  set(CMAKE_CUDA_FLAGS_DEBUG   "-O2 -Xcompiler='-ftree-vectorize -fopt-info-vec-optimized' -g")

  # warning suppression in release build (due to recursive kernel)
  set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -Xcompiler='-ftree-vectorize -mavx' -Xptxas='-suppress-stack-size-warning'")
  
  # build cuda lib
  enable_language(CUDA)
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

#### build/find dependencies ####
include_directories(
  libs/imgui                # Dear ImGui
  libs/imgui/backends
  libs/imgui/misc/freetype
  libs/stb                  # stb (image)
  )
add_subdirectory(libs)


find_package(OpenGL REQUIRED)              # opengl
include_directories(${OPENGL_INCLUDE_DIRS})
link_libraries(${OPENGL_LIBRARY_DIRS})
set(GLEW_LIBRARIES GLEW)                   # glew
set(GLFW_LIBRARIES glfw3)                  # glfw
find_package(nlohmann_json 3.2.0 REQUIRED) # json

# OS-dependent
set(EXTRA_LIBS "")
if (UNIX) # Linux (GTK)
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
message(STATUS " === CUDA LIBS: ==> ${CUDA_LIBRARIES}")
message(STATUS " === GLEW LIBS: ==> ${GLEW_LIBRARIES}")
message(STATUS " === GLFW LIBS: ==> ${GLFW_LIBRARIES}")
message(STATUS " === GL LIBS:   ==> ${OPENGL_LIBRARIES}")
message(STATUS " ===")
message(STATUS " ==")
message(STATUS " =")

# link executable
add_executable(${TARGET}
  src/cuda-tools.cpp
  src/image.cpp
  src/keyManager.cpp
  src/main.cpp
  src/settingForm.cpp
  src/simWindow.cpp
  src/tabMenu.cpp
  )
# main target compiler options
set(CMAKE_CXX_FLAGS_GLOBAL ) #-Werror=class-memaccess) #-Wall)
if ("${CMAKE_BUILD_TYPE}" MATCHES     "Debug")   # debug flags
  target_compile_options(${TARGET} PRIVATE ${CMAKE_CXX_FLAGS_GLOBAL} -O2 -ftree-vectorize -fopt-info-vec-optimized -g)
  
elseif ("${CMAKE_BUILD_TYPE}" MATCHES "Release") # release flags
  target_compile_options(${TARGET} PRIVATE ${CMAKE_CXX_FLAGS_GLOBAL} -O3 -ftree-vectorize -mavx)
  
endif ()

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
