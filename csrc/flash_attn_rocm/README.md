Here is the folder for APIs on rocm, which the backend code is from composable kernel.
Below is the introduction to the files.
"src/fmha.h" is the header file for the C++ APIs, in which declared the api function "run_fmha_fp16_gfx90a".
"fmha_api.cpp" is the c++ file that defined the function "run_fmha_fp16_gfx90a".
"src/fmha_fprop_fp16_kernel.gfx90a.cpp" is the interface that link API in fmha_api.cpp and the CK backend. 
"example_main.cpp" is an example which contains main function to test this API.
"compile.sh" is a compile script to compile the example above.