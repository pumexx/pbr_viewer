cmake_minimum_required(VERSION 3.7.0 FATAL_ERROR)
project( PBRVIEWER )

set( CMAKE_CXX_STANDARD 14 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

list( APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" )

include( pumex_macros )

set( PBRVIEWER_SHADER_NAMES
  data/shaders/direct_pbr.vert
  data/shaders/direct_pbr.frag
)

process_shaders( ${CMAKE_CURRENT_LIST_DIR} PBRVIEWER_SHADER_NAMES PBRVIEWER_INPUT_SHADERS PBRVIEWER_OUTPUT_SHADERS )
add_custom_target ( pbrviewer-shaders DEPENDS ${PBRVIEWER_OUTPUT_SHADERS} SOURCES ${PBRVIEWER_INPUT_SHADERS})

find_package( Vulkan REQUIRED )
find_package( pumex REQUIRED )

add_executable( pbrviewer pbrviewer.cpp )
target_include_directories( pbrviewer PUBLIC ${Vulkan_INCLUDE_DIRS} )
target_link_libraries( pbrviewer pumex::pumexlib )
add_dependencies( pbrviewer pbrviewer-shaders )
set_target_properties( pbrviewer PROPERTIES RESOURCE "${PBRVIEWER_OUTPUT_SHADERS}" )
set_target_postfixes( pbrviewer )

set_property(DIRECTORY ${PROJECT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT pbrviewer)
