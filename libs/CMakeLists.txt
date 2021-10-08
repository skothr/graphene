cmake_minimum_required(VERSION 3.16)

#### dependencies
# freetype (for better font rendering/hinting)
find_package(Freetype REQUIRED)

#### stb
include_directories(stb) # (header-only)
#### imgui
include_directories(imgui imgui/backends imgui/misc/freetype)
include_directories(../inc) # for vector.hpp (implicit type conversion in imconfig.h: Vec2f <-- ImVec2, etc.)
add_library(imgui STATIC
  imgui/imgui.cpp
  imgui/imgui_demo.cpp
  imgui/imgui_draw.cpp
  imgui/imgui_tables.cpp
  imgui/imgui_widgets.cpp
  imgui/backends/imgui_impl_glfw.cpp
  imgui/backends/imgui_impl_opengl3.cpp
  imgui/misc/freetype/imgui_freetype.cpp
  )
target_compile_definitions(imgui PRIVATE GLEW_STATIC IMGUI_IMPL_OPENGL_LOADER_GLEW)

set_target_properties (imgui PROPERTIES CXX_STANDARD 17)

if("${CMAKE_BUILD_TYPE}" MATCHES "Debug")
  target_compile_options(imgui PRIVATE -static -O0)
  target_link_options   (imgui PRIVATE -static -O0)
else()
  target_compile_options(imgui PRIVATE -static -O3 -ftree-vectorize -mavx -flto)
  target_link_options   (imgui PRIVATE -static -O3 -ftree-vectorize -mavx -flto)
endif()

target_include_directories(imgui PRIVATE ${FREETYPE_INCLUDE_DIRS})
target_link_libraries     (imgui ${FREETYPE_LIBRARIES})
