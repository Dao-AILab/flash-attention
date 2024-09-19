import json
from pathlib import Path

def write_file():
    TEMPLATE_PRELUDE = """#pragma once

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```
"""

    with open('headdim.json', 'r') as file:
        read_list = json.load(file)

    read_list += [
        [32,32],
        [64,64],
        [96,96],
        [128,128],
        [160,160],
        [192,192],
        [256,256],
    ]

    read_list = sorted(read_list, key=lambda x: (x[0], x[1]))

    TEMPLATE_BEGIN = """
#define QKHEADDIM_VHEADDIM_SWITCH(QKHEADDIM, VHEADDIM, ...)   \\
  [&] {                                                        \\
"""

    TEMPLATE_BODY = ""

    for qkhead_dim, vhead_dim in read_list[:-1]:
        TEMPLATE_BODY += f"""if (QKHEADDIM <= {qkhead_dim} && VHEADDIM <= {vhead_dim}) {{  \\
        constexpr static int kQKHeadDim = {qkhead_dim};                    \\
        constexpr static int kVHeadDim = {vhead_dim};                   \\
        return __VA_ARGS__();                                    \\
    }} else """

    qkhead_dim, vhead_dim = read_list[-1]
    TEMPLATE_BODY += f"""if (QKHEADDIM <= {qkhead_dim} && VHEADDIM <= {vhead_dim}) {{  \\
        constexpr static int kQKHeadDim = {qkhead_dim};                    \\
        constexpr static int kVHeadDim = {vhead_dim};                   \\
        return __VA_ARGS__();                                    \\
    }}                                                       \\
"""

    TEMPLATE_END = """}()
"""

    TEMPLATE = TEMPLATE_PRELUDE + TEMPLATE_BEGIN + TEMPLATE_BODY + TEMPLATE_END

    # print(TEMPLATE)
    with open(Path(__file__).parent.joinpath('static_switch_headdim.h'), 'w') as file:
        file.write(TEMPLATE)

if __name__ == '__main__':
    write_file()
