cmake_minimum_required(VERSION 3.6)
include_directories(.)
include_directories(../../inc) # for vector.hpp (implicit type conversion in imconfig.h: Vec2f <-- ImVec2, etc.)
add_library(imgui STATIC
  imgui.cpp
  imgui_impl_glfw.cpp
  imgui_impl_opengl3.cpp
  imgui_demo.cpp
  imgui_draw.cpp
  imgui_tables.cpp
  imgui_widgets.cpp
  ImGuiFileBrowser.cpp
  )
set_target_properties(imgui PROPERTIES CXX_STANDARD 17)
target_compile_definitions(imgui PRIVATE GLEW_STATIC IMGUI_IMPL_OPENGL_LOADER_GLEW) # FLOOR_IMGUI_POSITIONS=0)
target_compile_options(imgui PRIVATE -O2)
