cmake_minimum_required(VERSION 3.16)

project(graphene VERSION 0.1.2)
configure_file(inc/version/version.hpp.in ../inc/version/version.hpp)

set(TARGET graphene)
set(GRPH_ENABLE_CUDA TRUE)                        # set to TRUE to enable CUDA
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.2") # set to location of CUDA source
set(GRPH_CUDA_ARCH        "sm_75")                # set for specific graphics card

# set(BUILD_SHARED_LIBS 0)

# verbose Makefile for Debug build
if("${CMAKE_BUILD_TYPE}" MATCHES "Debug") # verbose build for debug
  set(CMAKE_VERBOSE_MAKEFILE 1)
endif()

include_directories(
  inc/
  inc/version
  )

#### build CUDA components ####
if(GRPH_ENABLE_CUDA)
  set(CUDACXX "nvcc")
  set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
  set(CMAKE_CUDA_FLAGS_DEBUG   "-O0 --expt-relaxed-constexpr -g")
  # warning suppression in release build (due to recursive kernel)
  set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --expt-relaxed-constexpr -Xcompiler='-ftree-vectorize -mavx' -Xptxas='-suppress-stack-size-warning'")
  
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
    src/cuda/fluid.cu
    src/cuda/maxwell.cu
    src/cuda/render.cu
    )
  set_target_properties(grphcuda PROPERTIES CXX_STANDARD 20)
  set(CUDA_LIBRARIES grphcuda)
endif()

#### build dependencies from source ####
include_directories(
  libs
  libs/imgui                # Dear ImGui
  libs/imgui/backends
  libs/imgui/misc/freetype
  libs/stb                  # stb (image)
  )
add_subdirectory(libs)
set(EXTRA_LIBS ${EXTRA_LIBS} imgui)

#### find external dependencies ####
find_package(OpenGL REQUIRED)              # opengl
include_directories(${OPENGL_INCLUDE_DIRS})
link_libraries(${OPENGL_LIBRARY_DIRS})
find_package(GLEW REQUIRED)                # GLEW
set(GLEW_LIBRARIES GLEW::GLEW)
find_package(glfw3 3.3 REQUIRED)           # glfw
set(GLFW_LIBRARIES glfw)
find_package(nlohmann_json 3.2.0 REQUIRED) # json
set(EXTRA_LIBS ${EXTRA_LIBS} nlohmann_json::nlohmann_json)

# OS-dependent
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
  src/cuda-tools.cpp
  src/image.cpp
  src/keyManager.cpp
  src/main.cpp
  src/settingForm.cpp
  src/simWindow.cpp
  src/tabMenu.cpp
  )
set_target_properties (${TARGET} PROPERTIES CXX_STANDARD 20)
target_compile_definitions(${TARGET} PRIVATE GLEW_STATIC)

# main target compiler options
set(CMAKE_CXX_FLAGS_GLOBAL "") # -static) #-Werror=class-memaccess) #-Wall) # TODO: Wall/Werror?
if     ("${CMAKE_BUILD_TYPE}" MATCHES "Debug")  # debug flags
  target_compile_options(${TARGET} PRIVATE ${CMAKE_CXX_FLAGS_GLOBAL} -O0 -g)
  target_link_options   (${TARGET} PRIVATE ${CMAKE_CXX_FLAGS_GLOBAL} -O0 -g)
elseif ("${CMAKE_BUILD_TYPE}" MATCHES "Release") # release flags
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
