Br = 64
Bc = 128
QKHeaddim = 128
VHeaddim = 256
smem =2 *(Br * QKHeaddim * 2  +  Br * VHeaddim  + Bc * QKHeaddim  + Bc * VHeaddim + Br * Bc * 2)
smem = smem/1024
print(smem)