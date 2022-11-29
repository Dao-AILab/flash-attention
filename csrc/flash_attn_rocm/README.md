Here is the folder for APIs on rocm, which the backend code is from composable kernel.

Below is the introduction to the files.

"src/fmha.h" is the header file for the C++ APIs, in which declared the  function "run_fmha_fp16_gfx90a".

"fmha_api.cpp" is the c++ file that defined the API function "mha_fwd", this function will call function "run_fmha_fp16_gfx90a". This function also contains a main function to test with the API.

"src/fmha_fprop_fp16_kernel.gfx90a.cpp" is the interface that link API in fmha_api.cpp and the CK backend, which defined function "run_fmha_fp16_gfx90a". In this function, it will use parameters conveyed from "mha_fwd" to initialize instance in CK and call CK function. Things still need to be done in this file is to find out and choose proper instance parameters according to the parameters from "mha_fwd".

"compile.sh" is a compile script to compile the example above.