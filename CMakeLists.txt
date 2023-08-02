cmake_minimum_required(VERSION 3.0.0)
project(RAI VERSION 0.1.0)

include(CTest)
enable_testing()

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

#set(CPACK_PROJECT_NAME ${PROJECT_NAME})
#set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
#include(CPack)
