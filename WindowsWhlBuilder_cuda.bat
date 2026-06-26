@echo off
rem =============================================================================
rem FlashAttention 2.8.4 (official + PyTorch2.10/cu132 fork) Windows wheel builder.
rem - Uses FLASH_ATTENTION_FORCE_BUILD=TRUE: always compiles from source.
rem - Run with the SAME python.exe you use at runtime (e.g. ComfyUI python_embeded).
rem Optional args (repeatable): FORCE_CXX11_ABI TRUE|FALSE   CUDA_ARCH 80;90;100;120
rem For torch 2.12+cu132, set CUDA_HOME to CUDA toolkit 13.2.
rem =============================================================================
setlocal enabledelayedexpansion

set MAX_JOBS=4

:parseArgs
if "%~1" == "FORCE_CXX11_ABI" (
    set "FLASH_ATTENTION_FORCE_CXX11_ABI=%~2"
    shift & shift
    goto :parseArgs
)
if "%~1" == "CUDA_ARCH" (
    set "FLASH_ATTN_CUDA_ARCHS=%~2"
    shift & shift
    goto :parseArgs
)
goto :buildContinue

:buildFinalize
set MAX_JOBS=
set BUILD_TARGET=
set DISTUTILS_USE_SDK=
set FLASH_ATTENTION_FORCE_BUILD=
set FLASH_ATTENTION_FORCE_CXX11_ABI=
set dist_dir=
set FLASH_ATTN_CUDA_ARCHS=
set tmpname=
endlocal
goto :eof

:buildContinue
echo MAX_JOBS: %MAX_JOBS%
echo FLASH_ATTENTION_FORCE_CXX11_ABI: %FLASH_ATTENTION_FORCE_CXX11_ABI%
echo FLASH_ATTN_CUDA_ARCHS: %FLASH_ATTN_CUDA_ARCHS%
pip install "setuptools>=49.6.0" packaging wheel psutil
set FLASH_ATTENTION_FORCE_BUILD=TRUE
set BUILD_TARGET=cuda
set DISTUTILS_USE_SDK=1
set dist_dir=dist
rem Default arch list excludes Thor sm_110 (see setup.py FORK_SUPPORTED_CUDA_ARCHS).
if not defined FLASH_ATTN_CUDA_ARCHS set FLASH_ATTN_CUDA_ARCHS=80;90;100;120

python setup.py bdist_wheel --dist-dir=%dist_dir%

rem rename whl — tag matches installed torch (e.g. cu132torch2.12.0 for CUDA 13.2 + torch 2.12.0)
for /f "delims=" %%i in ('python -c "import sys; from packaging.version import parse; import torch; python_version = f'cp{sys.version_info.major}{sys.version_info.minor}'; cxx11_abi=str(torch._C._GLIBCXX_USE_CXX11_ABI).upper(); torch_cuda_version = parse(torch.version.cuda); cuda_version = \"\".join(map(str, torch_cuda_version.release)); torch_version_raw = parse(torch.__version__); torch_version = \".\".join(map(str, torch_version_raw.release)); wheel_filename = f'cu{cuda_version}torch{torch_version}cxx11abi{cxx11_abi}'; print(wheel_filename);"') do set wheel_filename=%%i

set tmpname=%wheel_filename%

for %%i in (%dist_dir%\*.whl) do (
    set "filename=%%~nxi"
    echo !filename! | findstr /c:+ >nul
    if errorlevel 1 (
        set "count=0"
        for /l %%j in (0, 1, 1000) do (
            if "!filename:~%%j,1!"=="-" set /a count+=1
            if "!filename:~%%j,1!"=="-" if "!count!"=="2" (
                set "new_filename=!filename:~0,%%j!+%tmpname%!filename:~%%j!"
                echo Renaming !filename! to !new_filename!
                move "%%i" "!dist_dir!/!new_filename!"
                goto :next
            )
        )
    )
    :next
)

goto :buildFinalize
