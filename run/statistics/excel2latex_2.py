from xlwt import Workbook
import xlrd
import os


def read_excel(res_type):
    workbook = xlrd.open_workbook('C:/Users/SL/Downloads/VLDB_exp.xlsx')
    sheet = workbook.sheet_by_name('conventional_methods')
    results = []
    for i in range(4, 16):
        res = [sheet.cell(i, 18).value,
               sheet.cell(i, 19).value,
               sheet.cell(i, 20).value,
               sheet.cell(i, 21).value,
               sheet.cell(i, 22).value,
               sheet.cell(i, 23).value]

        res += [sheet.cell(i+17, 18).value,
                sheet.cell(i+17, 19).value,
                sheet.cell(i+17, 20).value,
                sheet.cell(i+17, 21).value,
                sheet.cell(i+17, 22).value,
                sheet.cell(i+17, 23).value]

        res += [sheet.cell(i + 17*2, 18).value,
                sheet.cell(i + 17*2, 19).value,
                sheet.cell(i + 17*2, 20).value,
                sheet.cell(i + 17*2, 21).value,
                sheet.cell(i + 17*2, 22).value,
                sheet.cell(i + 17*2, 23).value]

        res += [sheet.cell(i + 17 * 3, 18).value,
                sheet.cell(i + 17 * 3, 19).value,
                sheet.cell(i + 17 * 3, 20).value,
                sheet.cell(i + 17 * 3, 21).value,
                sheet.cell(i + 17 * 3, 22).value,
                sheet.cell(i + 17 * 3, 23).value]
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


def print_latex(res, dataset):
    print('\hline \parbox[t]{2mm}{\multirow{3}{*}{\\rotatebox[origin=c]{90}{' + dataset + '}}}')
    for i in range(3):
        m = 'LogMap'
        if i == 1:
            m = 'AML'
        elif i == 2:
            m = 'OpenEA'
        output = ' & ' + m
        j = 0
        if i == 2:
            while j < len(res[i]):
                output += ' & ${'+format_output(res[i][j])+'_{\,\pm\, '+format_output(res[i][j+1])+'}}^{*}$'
                j += 2
        else:
            while j < len(res[i]):
                output += ' & $'+format_output(res[i][j])+'_{\,\pm\, '+format_output(res[i][j+1])+'}$'
                j += 2
        print(output)
        print('\\\\')


def generate_latex(results):
    print(len(results))
    datasets = ['En-Fr', 'En-De', 'D-W', 'D-Y']
    print_latex(results[3: 6], datasets[0])
    print_latex(results[0: 3], datasets[1])
    print_latex(results[6: 9], datasets[2])
    print_latex(results[9: 12], datasets[3])


if __name__ == '__main__':
    generate_latex(read_excel('others'))
    # read_excel('others')
