# build dependencies
cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_FLAGS_DEBUG_INIT   "-O0 -Wall -static -g -pg")           # debug
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O3 -Wall -static -ftree-vectorize") # release

#### stb
include_directories(stb) # (header-only)

# freetype (for better font rendering/hinting)
find_package(Freetype REQUIRED)

#### imgui
include_directories(imgui)
include_directories(imgui/backends)
include_directories(imgui/misc/freetype)
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
set_target_properties(imgui PROPERTIES CXX_STANDARD 17)
target_compile_definitions(imgui PRIVATE GLEW_STATIC IMGUI_IMPL_OPENGL_LOADER_GLEW)
target_compile_options(imgui PRIVATE -O1)
target_link_libraries(imgui ${FREETYPE_LIBRARIES})
target_include_directories(imgui PRIVATE ${FREETYPE_INCLUDE_DIRS})
