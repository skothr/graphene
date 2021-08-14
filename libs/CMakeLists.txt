cmake_minimum_required(VERSION 3.6)
add_compile_options(-static -Wno-deprecated)

add_subdirectory(imgui)
# add_subdirectory(glew-2.1.0/build/cmake)
# add_subdirectory(glfw)
include_directories(stb)
# # JSON 
# set(JSON_BuildTests OFF CACHE INTERNAL "")
# set(JSON_Install OFF CACHE INTERNAL "")
# add_subdirectory(json)
