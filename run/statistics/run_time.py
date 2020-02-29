import os


def extract_results_from_each_line(folder, data_size):
    assert os.path.exists(folder)
    files = set()
    for file_folder in list(os.walk(folder))[1:]:
        for file in file_folder[2]:
            if data_size in file:
                files.add(file_folder[0]+'/'+file)

    files = sorted(files)
    result_temp = {}
    for file in files:
        time_sum = 0
        last1, last2 = 0, 0
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                time_item = 0
                if line.startswith('Training ends. Total time = '):
                    time_sum = float(line.split(' ')[-2])
                    last1 = 0
                    last2 = 0
                    break
                if 'time' in line:
                    time_item = float(line.strip('s\n').strip(' ').split('time')[-1].lstrip(' = ').lstrip(': ').strip(' s '))
                if 'entities costs' in line:
                    time_item = float(line.strip(' s.\n').split('entities costs ')[-1].strip(' s. '))
                time_sum += time_item
                last2 = last1
                last1 = time_item
        result_temp[file.split('/')[-1]] = time_sum - last1 - last2
    result_order_list = list(sorted(result_temp))
    results = []
    if len(result_temp) != 20:
        return [[0 for _ in range(5)] for _ in range(4)]
    i = 0
    while i < 20:
        result = []
        for _ in range(5):
            result.append(result_temp[result_order_list[i]])
            i += 1
        results.append(result)
    # print(results)
    return results


def extract_results(folder, data_size):
    assert os.path.exists(folder)
    files = set()
    for file_folder in list(os.walk(folder))[1:]:
        for file in file_folder[2]:
            if data_size in file:
                files.add(file_folder[0]+'/'+file)

    files = sorted(files)
    result_temp = {}
    for file in files:
        k = file.split('/')[-1]
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('Training ends. Total time = '):
                    result_temp[file.split('/')[-1]] = float(line.split(' ')[-2])
        if k not in result_temp:
            result_temp[k] = 0
    print(len(result_temp))
    result_order_list = list(sorted(result_temp))
    results = []
    if len(result_temp) != 20:
        return [[0 for _ in range(5)] for _ in range(4)]
    i = 0
    while i < 20:
        result = []
        for _ in range(5):
            result.append(result_temp[result_order_list[i]])
            i += 1
        results.append(result)
    #print(results)
    return results


def run(folder, data_size, method_list):
    assert os.path.exists(folder)
    res_dict = {}
    print(list(os.walk(folder))[0][1])
    for file_folder in list(os.walk(folder))[0][1]:
        if 'bootea' in file_folder or 'gcnalign' in file_folder:
            res_dict[file_folder] = extract_results_from_each_line(folder + file_folder + '/', data_size)
        else:
            res_dict[file_folder] = extract_results(folder+file_folder+'/', data_size)
    for i in range(4):
        for method in method_list:
            res = [0 for _ in range(5)]
            if method in res_dict:
                res = res_dict[method][i]
            print('%.3f\t%.3f\t%.3f\t%.3f\t%.3f' % (res[0], res[1], res[2], res[3], res[4]))


if __name__ == '__main__':
    data_size = '100K_V2'
    #method_list = ['mtranse', 'iptranse', 'jape', 'bootea', 'kdcoe', 'gcnalign', 'attre', 'imuse', 'sea', 'rsn4ea', 'multike']
    method_list = ['rsn4ea']
    # method_list = ['transh', 'transd', 'proje', 'conve', 'simple', 'rotate']
    run('/media/sl/Data/workspace/VLDB2020/output/log/', data_size, method_list)
    # extract_results_from_each_line('../../../output/log/bootea/', data_size)
