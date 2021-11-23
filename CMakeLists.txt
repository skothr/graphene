cmake_minimum_required(VERSION 3.16)

project(graphene VERSION 0.1.2)
configure_file(inc/version/version.hpp.in ../inc/version/version.hpp)

set(TARGET graphene)
set(GRPH_ENABLE_CUDA TRUE)                        # set to TRUE to enable CUDA
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.2") # set to location of CUDA source
set(GRPH_CUDA_ARCH        "sm_75")                # set for specific graphics card

# set(BUILD_SHARED_LIBS 0)

# verbose Makefile for debug build
if(("${CMAKE_BUILD_TYPE}" MATCHES "DEBUG") OR ("${CMAKE_BUILD_TYPE}" MATCHES "RELWITHDEBINFO")) # verbose builds
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
    src/cuda/em.cu
    src/cuda/render.cu
    src/cuda/vlines.cu
    )
  set_target_properties(grphcuda PROPERTIES CXX_STANDARD 17)
  set(CUDA_LIBRARIES grphcuda)
  
  
  if ("${CMAKE_BUILD_TYPE}" MATCHES "DEBUG") # DEBUG
    target_compile_options(grphcuda PRIVATE -arch ${GRPH_CUDA_ARCH} -O0 -g -pg --expt-relaxed-constexpr -use_fast_math)
    target_link_options   (grphcuda PRIVATE -arch ${GRPH_CUDA_ARCH} -O0 -g -pg)
  
  elseif ("${CMAKE_BUILD_TYPE}" MATCHES "RELWITHDEBINFO") # TEST
    target_compile_options(grphcuda PRIVATE -arch ${GRPH_CUDA_ARCH} -O2 -pg    --expt-relaxed-constexpr -use_fast_math -Xcompiler="-ftree-vectorize" -Xcompiler="-mavx")
    target_link_options   (grphcuda PRIVATE -arch ${GRPH_CUDA_ARCH} -O2 -pg)
   
  elseif ("${CMAKE_BUILD_TYPE}" MATCHES "RELEASE") # RELEASE
    target_compile_options(grphcuda PRIVATE -arch ${GRPH_CUDA_ARCH} -O3        --expt-relaxed-constexpr -use_fast_math
      -Xcompiler="-ftree-vectorize" -Xcompiler="-mavx" -Xptxas="-suppress-stack-size-warning") #-dlto
    target_link_options   (grphcuda PRIVATE -arch ${GRPH_CUDA_ARCH} -O3) # -dlto)
  endif()  
endif()

#### build dependencies from source ####
include_directories(
  libs
  libs/imgui                # Dear ImGui
  libs/imgui/backends
  libs/imgui/misc/freetype
  libs/stb                  # stb (image)
  libs/nlohmann             # nlohmann/json
  )
add_subdirectory(libs)
set(EXTRA_LIBS ${EXTRA_LIBS} imgui)

#### find external dependencies ####
find_package(OpenGL REQUIRED)              # opengl
include_directories(${OPENGL_INCLUDE_DIRS})
link_libraries(${OPENGL_LIBRARY_DIRS})
find_package(GLEW REQUIRED)                # GLEW
set(GLEW_LIBRARIES GLEW::GLEW)
set(GLFW_LIBRARIES glfw)                   # GLFW

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
  src/main.cpp
  src/cuda-tools.cpp
  src/image.cpp
  src/glShader.cpp
  src/keyFrameWidget.cpp
  src/keyBinding.cpp
  src/keyManager.cpp
  src/settingForm.cpp
  src/output.cpp
  src/tabMenu.cpp
  src/toolbar.cpp
  src/frameWriter.cpp
  src/simWindow.cpp
  )
set_target_properties (${TARGET} PROPERTIES CXX_STANDARD 20)
target_compile_definitions(${TARGET} PRIVATE GLEW_STATIC JSON_HAS_CPP_17)

# main target compiler options
set(CMAKE_CXX_FLAGS_GLOBAL "") # -static) #-Werror=class-memaccess) #-Wall) # TODO: Wall/Werror?
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
