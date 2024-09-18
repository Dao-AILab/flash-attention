Br = 128
Bc = 64
QKHeaddim = 128
VHeaddim = 256
bwdsmem =2 *(Br * QKHeaddim * 2  +  Br * VHeaddim  + Bc * QKHeaddim  + Bc * VHeaddim + Br * Bc * 2)
bwdsmem = bwdsmem/1024
fwdsmem = (Br * QKHeaddim  +  Bc * QKHeaddim  + Bc * VHeaddim)*2
fwdsmem = fwdsmem/1024
print("fwdsmem:", fwdsmem)
