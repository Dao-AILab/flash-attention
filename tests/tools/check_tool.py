import argparse
import numpy as np
import logging
import logging.handlers
from operator import itemgetter, attrgetter

logger = logging.getLogger("cheek-helper")
logger.setLevel(logging.DEBUG)

rf_handler = logging.StreamHandler()
rf_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s"))

logger.addHandler(rf_handler)

def get_meta_info(line):
    meta = line.strip().split(",")

    def _split(s):
        return [_s.strip() for _s in s.split("=")]

    meta_map = {}
    for item in meta:
        tmp = _split(item)
        meta_map[tmp[0]] = tmp[1]
    
    return meta_map["name"], int(meta_map["count"]), int(meta_map["lda"]), meta_map["position"], meta_map["type"]

def line_to_data(line):
    tmp = line.strip().split(",")

    def _process(info):
        return [float(i.strip()) for i in info[1:-1].split(":")]

    ret = []
    for item in tmp:
        if len(item) == 0:
            continue
        else:
            ret.append(_process(item.strip())[1]) 
    return ret

def get_array_from_file(file_name, tensor_name):
    ret = []
    name = ""
    count = 0
    lda = 0
    find = False
    meta_info = ""
    with open(file_name, 'r') as fp:
        for line in fp:
            line = line.strip()
            if line.strip().startswith("name"):
                name, count, lda, _, _ = get_meta_info(line)
                if name == tensor_name:
                    find = True
                    continue
            if find and line.startswith("["):
                ret.extend(line_to_data(line))
                if len(ret) == count:
                    break

    return np.array(ret)

def get_ele_from_line(line, key):
    index=0
    if key == "l":
        index = 26
    if line.find(key, index) != -1:
        val = line[line.find(key, index) : len(line)].split("=")[1].split(",")[0]
        # print(f" key = {key}, val={val}")
        return float(val)
    
    raise f"not find {key} in {line}"

def get_inf_id_from_line(line):
    def findall(string, s):
        ret = []
        index = 0

        while True:
            index = string.find(s, index)
            if index != -1:
                ret.append(index)
                index += len(s)
            else:
                break

        return ret

    all_pos = findall(line, "inf")

    id = []
    for pos in all_pos:
        id.append(int(line[pos - 6 : pos - 5]))
    return id

def is_same_matrix(test, groud_truth, label = "test", abs_eps = 0.01, relative_rps = 0.03):  
    logger.info("\n\n============= %s =============" % label)
    print("test shape is:", test.shape)  
    print("groud truth shape is:", groud_truth.shape)  
    # diff = np.abs(test - groud_truth) / np.abs(groud_truth)
    diff = np.abs(test - groud_truth)
    is_true = (diff < abs_eps).all()
    
    if not is_true:
        logger.warning(f"may have some ele which abs eps more than {abs_eps}, " 
                       f"please checkout final error_num, "
                       f"if not show, that means no ele whose relative eps more than {relative_rps}")
        error_num = 0
        for id, x in np.ndenumerate(diff):
            if x >= abs_eps:
                relative_diff = np.abs(x / test[id])
                if relative_diff > relative_rps:
                    print(id, x, test[id], relative_diff)
                    error_num = error_num + 1
        if error_num == 0:
            is_true = True
        else:
            print(f"error_num = {error_num}")
            print("test data is")
            print(test)
            print("groud truth data is")
            print(groud_truth)
            print("diff is")
            print(diff)
    print("\n================================================\n")

    print("%s" % ("passed" if is_true else "failed"))
    return is_true