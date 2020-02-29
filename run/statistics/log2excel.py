from xlwt import Workbook
import os


# FOLDER_PREFIX = '/media/sl/Data/workspace/VLDB2020/'
FOLDER_PREFIX = '/Users/sloriac/code/VLDB/'


def check_folder_path(path_str):
    assert path_str.startswith('results output folder:')
    print("results output folder:", path_str)
    folder_path = FOLDER_PREFIX + path_str.strip('\n').split('../../')[-1]
    if not os.path.exists(folder_path):
        print(folder_path)
    assert os.path.exists(folder_path)
    folder_path = folder_path.split('/2019')[0] + '/'
    folder_cnt = 0
    print("folder_path", folder_path)
    for _ in os.listdir(folder_path):
        if _ != ".DS_Store":
            folder_cnt = folder_cnt + 1
    if folder_cnt != 1:
        print(folder_path)
        print('Wrong:', folder_cnt, 'folders exist!')
    assert folder_cnt == 1


def judge(str1, str2):
    str1 = str1.strip('\n').split(', ')
    str2 = str2.strip('\n').split(', ')
    for i in range(len(str1)):
        if str1[i] != str2[i]:
            print(str1[i], '\t\t', str2[i])


def write2excel(result, book, sheet_name):
    sheet = book.add_sheet(sheet_name)
    sheet.write(1, 1, 'Dataset')
    sheet.write(1, 2, 'Hits1')
    sheet.write(1, 3, 'Hits5')
    sheet.write(1, 4, 'Hits10')
    sheet.write(1, 5, 'Hits50')
    sheet.write(1, 6, 'MR')
    sheet.write(1, 7, 'MRR')

    cnt_i, cnt_j = len(result)+1, 0
    for i in range(0, len(result)):
        for j in range(0, 7):
            sheet.write(i+2, j+1, result[i][j])
            if cnt_j == 0:
                sheet.write(cnt_i+2, cnt_j+1, result[i][j].split('_721_5fold_')[0])
                cnt_j += 1
            elif j != 0:
                sheet.write(cnt_i+2, cnt_j+1, result[i][j])
                cnt_j += 1
        if cnt_j == 31:
            cnt_j = 0
            cnt_i += 1


def extract_results(file_path, is_tune_param=False, is_csls=False):
    def extract_results_from_line(res_line):
        res = []
        res_line = res_line.strip('\n').split(' = ')
        hits = res_line[1].lstrip('[').split(']%')[0].split(' ')
        for hit in hits:
            if hit != '':
                res.append(float(hit)/100)
        res.append(float(res_line[2].split(',')[0]))
        res.append(float(res_line[3].split(',')[0]))
        return res

    prefix_str = 'accurate results'
    if is_csls:
        prefix_str = 'accurate results with csls'
    with open(file_path, 'r', encoding='utf-8') as file:
        # is_final_result = False
        # res_line_current = ''
        first_output_folder_line = True
        for line in file:
            if first_output_folder_line and line.startswith('results output folder'):
                check_folder_path(line)
                first_output_folder_line = False
            if line.startswith(prefix_str):
                res_line_current = line
                return extract_results_from_line(res_line_current)
            # if line.startswith('Training ends.') or 'should early stop' in line:
            #     if is_tune_param:
            #         return extract_results_from_line(res_line_current)
            #     is_final_result = True
            #     continue


def run(folder, model_name, is_csls=False, dataset_type='15K'):
    assert os.path.exists(folder)
    book = Workbook(encoding='utf-8')

    files = set()
    for file_folder in list(os.walk(folder))[1:]:
        for file in file_folder[2]:
            if dataset_type in file:
                files.add(file_folder[0]+'/'+file)

    files = sorted(files)

    result = []
    for file in files:
        row_name = file.split('/'+model_name+'_')[-1]
        res = [row_name]
        res.extend(extract_results(file, is_csls=is_csls))
        result.append(res)
    print(result)
    write2excel(result, book, model_name)
    file_name = folder + model_name + '_' + dataset_type
    if is_csls:
        file_name += '_csls'
    book.save(file_name+'.xlsx')


if __name__ == '__main__':
    method = 'sea'
    data_size = '100K'
    run('../../../output/log/'+method+'/', method, dataset_type=data_size)
    run('../../../output/log/'+method+'/', method, is_csls=True, dataset_type=data_size)
