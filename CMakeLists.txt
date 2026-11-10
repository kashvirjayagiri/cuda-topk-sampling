cmake_minimum_required(VERSION 3.24)
project(cuda_topk_sampling LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
find_package(CUDAToolkit REQUIRED)
add_library(full_sort STATIC src/full_sort.cu)
target_include_directories(full_sort PUBLIC include)
target_link_libraries(full_sort PUBLIC CUDA::cudart)
add_executable(test_full_sort tests/test_full_sort.cu)
target_link_libraries(test_full_sort PRIVATE full_sort)
enable_testing()
add_test(NAME full_sort_correctness COMMAND test_full_sort)

# A small shared-library bridge for loading saved PyTorch tensors from Python.
set_target_properties(full_sort PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(full_sort_python SHARED src/full_sort_api.cu)
target_link_libraries(full_sort_python PRIVATE full_sort)

add_library(hierarchical_topk STATIC src/hierarchical_topk.cu)
target_include_directories(hierarchical_topk PUBLIC include)
target_link_libraries(hierarchical_topk PUBLIC CUDA::cudart)
set_target_properties(hierarchical_topk PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(test_local_tile_topk tests/test_local_tile_topk.cu)
target_link_libraries(test_local_tile_topk PRIVATE hierarchical_topk)
add_test(NAME local_tile_topk_correctness COMMAND test_local_tile_topk)

add_library(local_tile_topk_python SHARED src/local_tile_topk_api.cu)
target_link_libraries(local_tile_topk_python PRIVATE hierarchical_topk)
