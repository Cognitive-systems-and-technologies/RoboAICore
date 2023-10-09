cmake_minimum_required(VERSION 3.17.0)
project(RAI VERSION 0.1.1)

include(CTest)
enable_testing()

option (RAI_CUDA "Use cuda compilation" OFF)
option (RAI_BUILD_ENVS "Build environments examples" ON)
if(RAI_CUDA)
	set (GPU_COMPUTE_CAPABILITY "60" CACHE STRING "")
endif()

SET (RL_SOURCE 
src/RL/ReplayBuffer.c
src/RL/SimpleDeque.c
src/RL/RLBrain.c
src/RL/RLAgent.c
src/RL/ACBrain.c
)
SET (RL_SOURCE_HEADERS 
src/RL/ReplayBuffer.h
src/RL/SimpleDeque.h
src/RL/RLBrain.h
src/RL/RLAgent.h
src/RL/ACBrain.h
)

SET (GEO_SOURCE 
src/geometry/TVec2.c
src/geometry/TVec3.c
src/geometry/TVec4.c

src/geometry/TRay.c
src/geometry/TQuaternion.c
src/geometry/TArc.c
)

SET (GEO_SOURCE_HEADERS 
src/geometry/TVec2.h
src/geometry/TVec3.h
src/geometry/TVec4.h

src/geometry/TRay.h
src/geometry/TQuaternion.h
src/geometry/TArc.h
)

SET (LIB_SOURCE 
src/Losses.c
src/Tensor.c
src/Tensor4.c
src/TWeightsInit.c
src/TCommon.c
src/TanhA.c
src/Relu.c
src/Optimizer.c
src/Model.c
src/MSE.c
src/Regression.c
src/MaxPool2d.c
src/Input.c
src/Dense.c
src/Conv2d.c
src/Utils.c
src/dList.c
src/tList.c
src/fList.c
src/Softmax.c

src/message.c
src/cJSON.c
src/cJSON_Utils.c
)

SET (LIB_CUDA_SOURCE 
  src/Input.cu
  src/Conv2d.cu
  src/Dense.cu
  src/MaxPool2d.cu
  src/Model.cu
  src/Relu.cu
  src/TanhA.cu
  src/Tensor.cu
  src/Tensor4.cu
  src/MSE.cu
  src/Optimizer.cu
  src/Utils.cu
  src/Losses.cu
)

SET (LIB_SOURCE_HEADERS 
src/Losses.h
src/Tensor.h
src/Tensor4.h
src/TWeightsInit.h
src/TCommon.h
src/TanhA.h
src/Relu.h
src/Optimizer.h
src/Model.h
src/MSE.h
src/Regression.h
src/MaxPool2d.h
src/Input.h
src/Dense.h
src/Conv2d.h
src/Utils.h
src/dList.h
src/tList.h
src/fList.h
src/Softmax.h

src/message.h
#src/stb_image.h
src/Interfaces.h
src/cJSON_Utils.h
src/cJSON.h
)

add_library(RAICore STATIC 
${LIB_SOURCE} 
${LIB_SOURCE_HEADERS}

${GEO_SOURCE} 
${GEO_SOURCE_HEADERS}

${RL_SOURCE} 
${RL_SOURCE_HEADERS}
)

if (RAI_CUDA)
  add_library(RAICoreGPU STATIC 
  ${LIB_SOURCE} 
  ${LIB_SOURCE_HEADERS}
  ${LIB_CUDA_SOURCE}

  ${GEO_SOURCE} 
  ${GEO_SOURCE_HEADERS}

  ${RL_SOURCE} 
  ${RL_SOURCE_HEADERS}
  )
  set_target_properties(RAICoreGPU PROPERTIES CUDA_ARCHITECTURES ${GPU_COMPUTE_CAPABILITY})
  target_include_directories (RAICoreGPU PUBLIC
  src
  )
endif()

#TODO: CUDA COMPILATION

target_include_directories (RAICore PUBLIC
  src
)

add_executable(data_test src/cmd/data_test.cpp)
target_link_libraries(data_test PUBLIC RAICore)

add_executable(rand_test src/cmd/rand_test.cpp)
target_link_libraries(rand_test PUBLIC RAICore)

add_executable(model_test src/cmd/model_test.cpp)
target_link_libraries(model_test PUBLIC RAICore)

add_executable(opt_test src/cmd/opt_test.cpp)
target_link_libraries(opt_test PUBLIC RAICore)

add_executable(mult_opt_test src/cmd/mult_opt_test.cpp)
target_link_libraries(mult_opt_test PUBLIC RAICore)

if (RAI_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit)
    add_executable(cuda_test src/cmd/cuda_test.cu)
    target_link_libraries(cuda_test PUBLIC RAICoreGPU CUDA::cudart CUDA::cuda_driver)
    set_target_properties(cuda_test PROPERTIES CUDA_ARCHITECTURES ${GPU_COMPUTE_CAPABILITY})
  else()
    message(STATUS "No CUDA compiler found")
  endif()
endif ()

if(RAI_BUILD_ENVS)
    add_subdirectory (${CMAKE_CURRENT_SOURCE_DIR}/ext)
    ###############################
    #qmaze example
    add_executable(qmaze src/cmd/qmaze/qmaze.cpp 
    src/cmd/qmaze/agent.h 
    src/cmd/qmaze/cell.h 
    src/cmd/qmaze/grid.h 
    src/cmd/qmaze/quad.h)

    target_link_libraries(qmaze PUBLIC RAICore glfw glad)
    target_include_directories (qmaze PUBLIC
      src
      src/cmd/qmaze
      ${GLFW_DEPS}
      ${GLFW_INCLUDE}
      ${GLAD_INCLUDE}
    )

    ###############################
    #cartpole example
    add_executable(cartpole src/cmd/cartpole/cartpole.cpp 
    src/cmd/cartpole/shapes.h
    src/cmd/cartpole/cart.h
    src/cmd/cartpole/agent.h)

    target_link_libraries(cartpole PUBLIC RAICore glfw glad box2d)
    target_include_directories (cartpole PUBLIC
      src
      src/cmd/cartpole
      ${GLFW_DEPS}
      ${GLFW_INCLUDE}
      ${GLAD_INCLUDE}
      ${BOX2D_INCLUDE}
    )
endif ()

#set(CPACK_PROJECT_NAME ${PROJECT_NAME})
#set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
#include(CPack)
