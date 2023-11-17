// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/host_utility/hip_check_error.hpp"

#include "ck/library/utility/device_memory.hpp"

DeviceMem::DeviceMem(std::size_t mem_size) : mMemSize(mem_size)
{
    hip_check_error(hipMalloc(static_cast<void**>(&mpDeviceBuf), mMemSize));
}

void DeviceMem::Realloc(std::size_t mem_size)
{
    if(mpDeviceBuf)
    {
        hip_check_error(hipFree(mpDeviceBuf));
    }
    mMemSize = mem_size;
    hip_check_error(hipMalloc(static_cast<void**>(&mpDeviceBuf), mMemSize));
}

void* DeviceMem::GetDeviceBuffer() const { return mpDeviceBuf; }

std::size_t DeviceMem::GetBufferSize() const { return mMemSize; }

void DeviceMem::ToDevice(const void* p) const
{
    if(mpDeviceBuf)
    {
        hip_check_error(
            hipMemcpy(mpDeviceBuf, const_cast<void*>(p), mMemSize, hipMemcpyHostToDevice));
    }
    else
    {
        throw std::runtime_error("ToDevice with an empty pointer");
    }
}

void DeviceMem::FromDevice(void* p) const
{
    if(mpDeviceBuf)
    {
        hip_check_error(hipMemcpy(p, mpDeviceBuf, mMemSize, hipMemcpyDeviceToHost));
    }
    else
    {
        throw std::runtime_error("FromDevice with an empty pointer");
    }
}

void DeviceMem::SetZero() const
{
    if(mpDeviceBuf)
    {
        hip_check_error(hipMemset(mpDeviceBuf, 0, mMemSize));
    }
}

DeviceMem::~DeviceMem()
{
    if(mpDeviceBuf)
    {
        hip_check_error(hipFree(mpDeviceBuf));
    }
}
