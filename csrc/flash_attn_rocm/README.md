Here is the folder for APIs on rocm, which the backend code is from composable kernel.

Below is the introduction to the files.

"src/fmha.h" is the header file for the C++ APIs, in which declared the  function "run_fmha_fp16_gfx90a".

"fmha_api.cpp" is the c++ file that defined the API function "mha_fwd", this function will call function "run_fmha_fp16_gfx90a". This function also contains a main function to test with the API.

"src/fmha_fprop_fp16_kernel.gfx90a.cpp" is the interface that link API in fmha_api.cpp and the CK backend, which defined function "run_fmha_fp16_gfx90a". In this function, it will use parameters conveyed from "mha_fwd" to initialize instance in CK and call CK function. Things still need to be done in this file is to find out and choose proper instance parameters according to the parameters from "mha_fwd".

"CMakeList.txt" is a cmake file to compile the example above.

Useage for "CMakeLists.txt": 
```
$mkdir build
$cd build
$cmake ..
$make
```

My docker is from https://hub.docker.com/layers/rocm/pytorch/rocm5.3.2_ubuntu20.04_py3.7_pytorch_1.12.1/images/sha256-387b2538d14cfd55a9510b7ea07049f1e71b7e755413080153b997c798fe5099?context=explore

If you choose another docker or you install pytorch by yourself.

Please change line 8 in CMakeLists.txt file with your own path.

You can use command
``` 
python -c 'import torch;print(torch.utils.cmake_prefix_path)'
```
to find your path.


