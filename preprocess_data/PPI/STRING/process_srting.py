import pandas as pd
import networkx as nx
import numpy as np

#####   运行后得到一个 序号，基因文件string_index_file.txt和 序号-序号的联系文件
#####   删除置信度小于0.7的，即联合分数700以下的
id_to_gene = {}
gene_list = []
index = 1
string_index = open('./string_index_file.txt','w',encoding='utf-8')
gene_to_xuhao = {}
with open('./9606.protein.info.v12.0.txt','r')as f:
    for li in f.readlines()[1:]: # 从第二行开始读取
        li = li.replace('\n','').split('\t')
        id_to_gene[li[0]] = li[1]
        string_index.write(str(index)+' '+li[1]+'\n')
        gene_to_xuhao[li[1]] = str(index)
        index += 1
string_index.close()

# id 转换并保存
string_edge = open('./string_edge_file.txt','w',encoding='utf-8')
string_edge_xuhao = open('./string_edge_xuhao_file.txt','w',encoding='utf-8')
with open('./9606.protein.links.full.v12.0.txt','r')as f:
    for li in f.readlines()[1:]: # 从第二行开始读取
        li = li.replace('\n','').split(' ')
        if int(li[-1]) >= 700:
            string_edge.write(id_to_gene[li[0]]+' '+id_to_gene[li[1]]+' '+li[-1]+'\n')
            string_edge_xuhao.write(gene_to_xuhao[id_to_gene[li[0]]]+' '+gene_to_xuhao[id_to_gene[li[1]]]+' '+li[-1]+'\n')
string_edge.close()
string_edge_xuhao.close()

string_edge = pd.read_csv('./string_edge_file.txt', encoding='utf-8', sep=' ')
string_edge1 = string_edge.iloc[:, :2]
string_edge1.to_csv('./string_edge.txt', sep='\t', index=False)




