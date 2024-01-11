cmake_minimum_required(VERSION 3.10)

add_definitions(-DLLAMA_DISABLE_LOGGING)

# Optional Development Team attribute
set(DEV_TEAM "" CACHE STRING "Xcode Development Team ID")
set(CMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM ${DEV_TEAM})
set(CMAKE_XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "Apple Development")
set(CMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED YES)

project(llm)
set(COMMON_SRC 
  src/llm.cpp
  # src/simple_map.c
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Platform-specific library and include directories
set(LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/${PLATFORM}")
set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Set the correct library and include paths
link_directories(${LIB_DIR})
include_directories(${INCLUDE_DIR})

find_package(Threads REQUIRED)

find_library(LLAMA_LIBRARY NAMES llama PATHS ${LIB_DIR} NO_DEFAULT_PATH)
find_library(COMMON_LIBRARY NAMES common PATHS ${LIB_DIR} NO_DEFAULT_PATH)
find_library(GGML_LIBRARY NAMES ggml PATHS ${LIB_DIR} NO_DEFAULT_PATH)

# Option to build the executable
option(BUILD_EXECUTABLE "Build the executable" OFF)
if(BUILD_EXECUTABLE)
  add_executable(llm_exec src/main.cpp ${COMMON_SRC})
  set_target_properties(llm_exec PROPERTIES OUTPUT_NAME llm)
  target_link_libraries(llm_exec PRIVATE ${CMAKE_THREAD_LIBS_INIT} ${COMMON_LIBRARY} ${GGML_LIBRARY} ${LLAMA_LIBRARY})
endif()

add_library(llm_lib SHARED ${COMMON_SRC})
set_target_properties(llm_lib PROPERTIES OUTPUT_NAME llm)
target_link_libraries(llm_lib PRIVATE ${CMAKE_THREAD_LIBS_INIT} ${COMMON_LIBRARY} ${GGML_LIBRARY} ${LLAMA_LIBRARY})

# Platform-specific configurations
if(APPLE)
  find_library(FOUNDATION_LIBRARY Foundation)
  find_library(METAL_FRAMEWORK Metal)
  find_library(METALKIT_FRAMEWORK MetalKit)
  find_library(ACCELERATE_LIBRARY Accelerate)
  
  if(NOT FOUNDATION_LIBRARY OR NOT METAL_FRAMEWORK OR NOT METALKIT_FRAMEWORK OR NOT ACCELERATE_LIBRARY)
    message(FATAL_ERROR "One or more macOS specific libraries were not found.")
  endif()

  add_compile_definitions(GGML_USE_METAL)
  target_link_libraries(llm_lib PRIVATE ${FOUNDATION_LIBRARY} ${METAL_FRAMEWORK} ${METALKIT_FRAMEWORK} ${ACCELERATE_LIBRARY})

  if(BUILD_EXECUTABLE)
    target_link_libraries(llm_exec PRIVATE ${FOUNDATION_LIBRARY} ${METAL_FRAMEWORK} ${METALKIT_FRAMEWORK} ${ACCELERATE_LIBRARY})
  endif()
else()
  find_package(CUDAToolkit)
  
  if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    add_compile_definitions(GGML_USE_CUBLAS)
    target_link_libraries(llm_lib PRIVATE CUDA::cublas cudart)
  else()
    message(FATAL_ERROR "CUDA Toolkit not found.")
  endif()
endif()

install (TARGETS llm_lib DESTINATION ${CMAKE_CURRENT_LIST_DIR}/install/${PLATFORM}/lib)
install (FILES src/llm.h DESTINATION ${CMAKE_CURRENT_LIST_DIR}/install/${PLATFORM}/include)
