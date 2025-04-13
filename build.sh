
export http_proxy="http://sys-proxy-rd-relay.byted.org:8118"
export https_proxy="http://sys-proxy-rd-relay.byted.org:8118"
export no_proxy="byted.org"

pip3 install --no-cache-dir --upgrade pip
pip3 install requests psutil ninja
pip3 install --no-cache-dir http://luban-source.byted.org/repository/scm/lab.pytorch.pytorch2_1.0.0.350.tar.gz

cd hopper
export MAX_JOBS=8 
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export LDFLAGS=-L/usr/local/cuda/lib64/stubs
python3 setup.py bdist_wheel
