cmake_minimum_required (VERSION 3.17)

###########################################################################
# box2d

set (BOX2D_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/box2d/include PARENT_SCOPE)
set (BOX2D_BUILD_TESTBED OFF CACHE BOOL "" FORCE)

add_subdirectory (box2d)

set_property (TARGET box2d PROPERTY FOLDER "ext")

###########################################################################
# glfw / glad

set (GLFW_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/glfw/include PARENT_SCOPE)
set (GLFW_DEPS ${CMAKE_CURRENT_SOURCE_DIR}/glfw/deps PARENT_SCOPE)
set (GLAD_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/glad/include PARENT_SCOPE)

set (GLFW_LIBRARY_TYPE STATIC CACHE STRING "" FORCE)
set (GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set (GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set (GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory (glfw)
add_subdirectory (glad)

set_property (TARGET glfw PROPERTY FOLDER "ext")
set_property (TARGET glad PROPERTY FOLDER "ext")

