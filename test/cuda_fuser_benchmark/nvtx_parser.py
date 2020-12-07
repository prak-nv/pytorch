def time_to_float(time):
    m = 1.
    if 'us' in time:
        time = float(time[:-2])
    elif 'ms' in time:
        time = float(time[:-2])
        m = 1000.
    else:
        time = float(time[:-1])
        m = 1000000.
    return m * time

def parse_result(lines, nvtx_mark):

    lines = lines.split('\n')

    result = []
    # parse result for kernel time
    for i, line in enumerate(lines):
        if '==' in line and nvtx_mark in line:
            # kernel time
            match = (line.split("\"")[1]).split("_")
            res = lines[i+3].split()
            percentage = float(res[2][:-1])/100
            time = time_to_float(res[3])
            kernel_time = time / percentage / 1000
            match[-1] = kernel_time
            result.append(match)

    return result
