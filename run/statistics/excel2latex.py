from xlwt import Workbook
import xlrd
import os


def read_excel(res_type):
    start_col = 6
    start_row = 2
    method_num = 11
    if res_type == 'others':
        start_col = 57
        method_num = 6
    elif res_type == 'main_csls':
        start_row = 29
    elif res_type == 'others_csls':
        start_col = 57
        start_row = 29
        method_num = 6
    workbook = xlrd.open_workbook('/home/sl/下载/VLDB_exp.xlsx')
    sheet = workbook.sheet_by_name('main_results')
    # print(sheet.name, sheet.nrows, sheet.ncols)

    results = []
    for col in range(start_col, start_col + method_num * 4):
        res = [sheet.cell(col, start_row).value,
               sheet.cell(col, start_row + 1).value, sheet.cell(col, start_row + 2).value,
               sheet.cell(col, start_row + 3).value, sheet.cell(col, start_row + 4).value,
               sheet.cell(col, start_row + 11).value, sheet.cell(col, start_row + 12).value,
               sheet.cell(col, start_row + 13).value, sheet.cell(col, start_row + 14).value,
               sheet.cell(col, start_row + 15).value, sheet.cell(col, start_row + 16).value,
               sheet.cell(col, start_row + 23).value, sheet.cell(col, start_row + 24).value]
        col_100K = col + 82
        res += [sheet.cell(col_100K, start_row + 1).value, sheet.cell(col_100K, start_row + 2).value,
                sheet.cell(col_100K, start_row + 3).value, sheet.cell(col_100K, start_row + 4).value,
                sheet.cell(col_100K, start_row + 11).value, sheet.cell(col_100K, start_row + 12).value,
                sheet.cell(col_100K, start_row + 13).value, sheet.cell(col_100K, start_row + 14).value,
                sheet.cell(col_100K, start_row + 15).value, sheet.cell(col_100K, start_row + 16).value,
                sheet.cell(col_100K, start_row + 23).value, sheet.cell(col_100K, start_row + 24).value]

        # print(res)
        results.append(res)
    # print(len(results))
    return results


# \hline \parbox[t]{2mm}{\multirow{11}{*}{\rotatebox[origin=c]{90}{DB\textsubscript{EN-DE}}}}

# & MTransE	& $.000_{\,\pm\, .000}$ & $.000_{\,\pm\, .000}$ & $.000_{\,\pm\, .000}$
# 			& $.000_{\,\pm\, .000}$ & $.000_{\,\pm\, .000}$ & $.000_{\,\pm\, .000}$
# 			& $.000_{\,\pm\, .000}$ & $.000_{\,\pm\, .000}$ & $.000_{\,\pm\, .000}$
# 			& $.000_{\,\pm\, .000}$ & $.000_{\,\pm\, .000}$ & $.000_{\,\pm\, .000}$ \\


def format_output(float_num):
    if float_num == '':
        return '.000'
    if float_num > 1:
        return str(round(float_num))
    float_num = str(round(float_num, 3))[1:]
    while len(float_num) < 4:
        float_num += '0'
    return float_num


def generate_latex(results):
    method_num = len(results) // 4
    datasets = ['DB\\textsubscript{EN-DE}', 'DB\\textsubscript{EN-FR}', 'DB-WD', 'DB-YG']
    for i in range(4):
        print('\hline \parbox[t]{2mm}{\multirow{'+str(method_num)+'}{*}{\\rotatebox[origin=c]{90}{'+datasets[i]+'}}}')
        for j in range(method_num):
            col = i*method_num + j
            for k in range(4):
                if k == 0:
                    print('& %s	& $%s_{\,\pm\, %s}$ & $%s_{\,\pm\, %s}$ & $%s_{\,\pm\, %s}$'
                          % (results[col][0], format_output(results[col][1]), format_output(results[col][2]),
                             format_output(results[col][3]), format_output(results[col][4]),
                             format_output(results[col][5]), format_output(results[col][6])))
                else:
                    temp = 4
                    if k == 1:
                        temp = 2
                    elif k == 3:
                        temp = 6
                    print('\t& $%s_{\,\pm\, %s}$ & $%s_{\,\pm\, %s}$ & $%s_{\,\pm\, %s}$'
                          % (format_output(results[col][k*4+1+temp]), format_output(results[col][k*4+2+temp]),
                             format_output(results[col][k*4+3+temp]), format_output(results[col][k*4+4+temp]),
                             format_output(results[col][k*4+5+temp]), format_output(results[col][k*4+6+temp])))
            print('\\\\')


def generate_latex_csls(results):
    method_num = len(results) // 4
    datasets = ['DB\\textsubscript{EN-FR}', 'DB\\textsubscript{EN-DE}', 'DB-WD', 'DB-YG']
    for i in range(4):
        print('\hline \parbox[t]{2mm}{\multirow{'+str(method_num)+'}{*}{\\rotatebox[origin=c]{90}{'+datasets[i]+'}}}')
        for j in range(method_num):
            col = i*method_num + j
            for k in range(4):
                if k == 0:
                    print('& %s	& $%s_{\,\pm\, %s}$ & $%s_{\,\pm\, %s}$ & $%s_{\,\pm\, %s}$'
                          % (results[col][0], format_output(results[col][1]), format_output(results[col][2]),
                             format_output(results[col][3]), format_output(results[col][4]),
                             format_output(results[col][5]), format_output(results[col][6])))
                else:
                    temp = 4
                    if k == 1:
                        temp = 2
                    elif k == 3:
                        temp = 6
                    print('\t& $%s_{\,\pm\, %s}$ & $%s_{\,\pm\, %s}$ & $%s_{\,\pm\, %s}$'
                          % (format_output(results[col][k*4+1+temp]), format_output(results[col][k*4+2+temp]),
                             format_output(results[col][k*4+3+temp]), format_output(results[col][k*4+4+temp]),
                             format_output(results[col][k*4+5+temp]), format_output(results[col][k*4+6+temp])))
            print('\\\\')


if __name__ == '__main__':
    generate_latex(read_excel('others'))
