import numpy as np
import sys
import xlrd
import xlwt
from xlutils.copy import copy
import os

def read_excel(path,sheet_index=0):                                                                                     #读取excel表格，将结果转为list返回
    workbook = xlrd.open_workbook(path)
    table = workbook.sheet_by_index(sheet_index)
    r = table.nrows
    c=table.ncols
    list=[[0 for col in range(c)]for row in range(r) ]
    for i in range(r):
        for j in range(c):
            list[i][j]=table.cell(i,j).value
    return list
def write_excel(list_content,path,sheet_index=0,bool_row_append=True,bool_written_in_row=True):                         #将list形式的数据写excel表格
    #list_content需要为二维列表
    #原文件存在则在其表后续写，否则创建新文件并写入
    #bool_row_append为true表示在最后一行下续写，false表示在最后一列右侧续写
    '''bool_written_in_row为true表示将list中的内容按行书写，false为按列书写
    ex. false时[[1,2,3][1,2]]
        [1,2,3]按照列写入表格
    '''
    if (os.path.exists(path)):
        workbook = xlrd.open_workbook(path)
        try:
            table = workbook.sheet_by_index(sheet_index)
            r = table.nrows
            c = table.ncols
            new_workbook = copy(workbook)
        except IndexError:
            #sheet doesn't exist
            r=0
            c=0
            new_workbook = copy(workbook)
            new_workbook.add_sheet(str(sheet_index))
        sheet = new_workbook.get_sheet(sheet_index)
    else:
        new_workbook = xlwt.Workbook()
        sheet = new_workbook.add_sheet("sheet1")
        r = 0
        c = 0
    if(bool_row_append):
        c=0                                                                                                             #在原文件后一行开始续写
    else:
        r=0
    if(bool_written_in_row):
        for i in range(len(list_content)):
            for j in range(len(list_content[i])):
                sheet.write(i + r, j+c, list_content[i][j])
    else:
        for i in range(len(list_content)):
            for j in range(len(list_content[i])):
                sheet.write(j+r,i+c,list_content[i][j])
    new_workbook.save(path)
