import pandas as pd
import mygene



pd.set_option('display.max_columns', None)
def get_gene_symbols(list_of_ensembl_ids):
    # get Ensembl IDs for gene names
    mg = mygene.MyGeneInfo()
    res = mg.querymany(list_of_ensembl_ids,
                       scopes='ensembl.gene',
                       fields='symbol',
                       species='human', returnall=True
                      )

    def get_symbol_and_ensembl(d):
        if 'symbol' in d:
            return [d['query'], d['symbol']]
        else:
            return [d['query'], None]

    node_names = [get_symbol_and_ensembl(d) for d in res['out']]
    # now, retrieve the names and IDs from a dictionary and put in DF
    node_names = pd.DataFrame(node_names, columns=['Ensembl_ID', 'Symbol']).set_index('Ensembl_ID')
    node_names.dropna(axis=0, inplace=True)
    return node_names


def get_gene_symbols_from_proteins(list_of_ensembl_ids):
    # get Ensembl IDs for gene names
    mg = mygene.MyGeneInfo()
    res = mg.querymany(list_of_ensembl_ids,
                       scopes='ensembl.protein',
                       fields='symbol',
                       species='human', returnall=True
                      )

    def get_symbol_and_ensembl(d):
        if 'symbol' in d:
            return [d['query'], d['symbol']]
        else:
            return [d['query'], None]

    node_names = [get_symbol_and_ensembl(d) for d in res['out']]
    # now, retrieve the names and IDs from a dictionary and put in DF
    node_names = pd.DataFrame(node_names, columns=['Ensembl_ID', 'Symbol']).set_index('Ensembl_ID')
    node_names.dropna(axis=0, inplace=True)
    return node_names


interactions = pd.read_csv('./ConsensusPathDB_human_PPI.gz',
                           compression='gzip',
                           header=1,
                           sep='\t',
                           encoding='utf8'
                          )
interactions_nona = interactions.dropna()

# select interactions with exactly two partners
binary_inter = interactions_nona[interactions_nona.interaction_participants__uniprot_entry.str.count(',') == 1] #interaction_participants__genename
# split the interactions columns into interaction partners
edgelist = pd.concat([binary_inter.interaction_participants__uniprot_entry.str.split(',', expand=True),  #interaction_participants__genename
                                binary_inter.interaction_confidence], axis=1
                              )
# make the dataframe beautiful
# edgelist.set_index([np.arange(edgelist.shape[0])], inplace=True)
edgelist.columns = ['partner1', 'partner2', 'confidence']
edgelist.to_csv('./CPDB_uni_edgelist_ALL.tsv', sep='\t')
print(edgelist)

# select interactions with confidence score above threshold
high_conf_edgelist = edgelist[edgelist.confidence > 0.5]
# high_conf_edgelist = high_conf_edgelist.replace('', np.nan)
# high_conf_edgelist = high_conf_edgelist.dropna()
# high_conf_edgelist = high_conf_edgelist.sort_values(by='partner1')
high_conf_edgelist.to_csv('./CPDB_uni_edgelist_0.5.tsv', sep='\t', index=False)
# high_conf_edgelist1 = high_conf_edgelist.sort_values(by='partner1')
# print(high_conf_edgelist1)
high_conf_edgelist1 = high_conf_edgelist.drop("confidence", axis=1)
high_conf_edgelist1.to_csv('./CPDB_uni_edgelist_0.5.txt', sep='\t', index=False, header=False)
print(high_conf_edgelist1)
print(high_conf_edgelist.head())



mapping = pd.read_csv('./idmapping0.5_2024_06_28.tsv',
                      sep='\t',
                      header=0,
                      names=['ensembl']
                     )

# get them into our dataframe (size increases because of duplicates in mapping)
# that is, one uniprot gene name has multiple ensembl gene names, hence we have to add those interactions
p1_incl = high_conf_edgelist.join(mapping, on='partner1', how='inner', rsuffix='_p1')
both_incl = p1_incl.join(mapping, on='partner2', how='inner', rsuffix='_p2')
both_incl.columns = ['partner1', 'partner2', 'confidence', 'partner1_ensembl', 'partner2_ensembl']
print(both_incl)


# collect statistics on how many interactions we lost
num_unmaps = both_incl[both_incl.partner1_ensembl.isnull() | both_incl.partner2_ensembl.isnull()].shape[0]
num_p1_unmaps = p1_incl[p1_incl.ensembl.isnull()].partner1.unique().shape[0]
num_p2_unmaps = both_incl[both_incl.partner2_ensembl.isnull()].partner2.unique().shape[0]
print ("We were unable to map {} source and {} target genes.".format(num_p1_unmaps, num_p2_unmaps))
print ("We lost {} interactions this way.".format(num_unmaps))

# kick out the NaNs and remove uniprot names
final_edgelist = both_incl.dropna(axis=0)
final_edgelist.drop(['partner1', 'partner2'], axis=1, inplace=True)
print ("Final edge list has {} interactions".format(final_edgelist.shape[0]))

# sort by number and put confidence at last and rename columns
final_edgelist.sort_index(inplace=True)
print(final_edgelist.head())
cols = final_edgelist.columns.tolist()
cols = cols[1:] + [cols[0]]
final_edgelist = final_edgelist[cols]
final_edgelist.columns = ['partner1', 'partner2', 'confidence']

# 去掉版本号
final_edgelist['partner1'] = final_edgelist['partner1'].str.split('.').str[0]
final_edgelist['partner2'] = final_edgelist['partner2'].str.split('.').str[0]
print(final_edgelist)

# write to file and look at the first rows
final_edgelist.sort_index(inplace=True)
final_edgelist.to_csv('./CPDB_ensg_edgelist_0.5.tsv', sep='\t')
# save_sif(final_edgelist, './CPDB_ensg_edgelist.sif')
print(final_edgelist.head())

# ens_names = final_edgelist.partner1.append(final_edgelist.partner2).unique()
ens_names = pd.concat([final_edgelist['partner1'], final_edgelist['partner2']]).unique()
ens_to_symbol = get_gene_symbols(ens_names)
print(ens_to_symbol)

p1_incl = final_edgelist.join(ens_to_symbol, on='partner1', how='inner', rsuffix='_p1')
both_incl = p1_incl.join(ens_to_symbol, on='partner2', how='inner', rsuffix='_p2')
both_incl.columns = ['partner1', 'partner2', 'confidence', 'partner1_symbol', 'partner2_symbol']
print(both_incl)

# collect statistics on how many interactions we lost
num_unmaps = both_incl[both_incl.partner1_symbol.isnull() | both_incl.partner2_symbol.isnull()].shape[0]
num_p1_unmaps = p1_incl[p1_incl.Symbol.isnull()].partner1.unique().shape[0]
num_p2_unmaps = both_incl[both_incl.partner2_symbol.isnull()].partner2.unique().shape[0]
print ("We were unable to map {} source and {} target genes.".format(num_p1_unmaps, num_p2_unmaps))
print ("We lost {} interactions this way.".format(num_unmaps))

# kick out the NaNs and remove ensembl IDs
final_edgelist_symbols = both_incl.dropna(axis=0)
final_edgelist_symbols.drop(['partner1', 'partner2'], axis=1, inplace=True)
print ("Final edge list has {} interactions".format(final_edgelist_symbols.shape[0]))

# sort by number and put confidence at last and rename columns
final_edgelist_symbols.sort_index(inplace=True)
cols = final_edgelist_symbols.columns.tolist()
cols = cols[1:] + [cols[0]]
final_edgelist_symbols = final_edgelist_symbols[cols]
final_edgelist_symbols.columns = ['partner1', 'partner2', 'confidence']

# remove duplicated interactions (can happen due to multiple Ensembl IDs mapping to the same gene name)
no_interactions = final_edgelist_symbols.shape[0]
final_edgelist_symbols.drop_duplicates(inplace=True)
print ("Dropping {} interactions because the are redundant for gene names".format(no_interactions - final_edgelist_symbols.shape[0]))

# write to file and look at the first rows
final_edgelist_symbols = final_edgelist_symbols.sort_values(by='partner1')
final_edgelist_symbols.to_csv('./CPDB_symbols_edgelist_0.5.tsv', sep='\t', index=False)
print(final_edgelist_symbols.head())
final_edgelist_symbols1 = final_edgelist_symbols.drop("confidence", axis=1)
final_edgelist_symbols1 = final_edgelist_symbols1.sort_values(by='partner1')
final_edgelist_symbols1.to_csv('./CPDB_symbols_edgelist_0.5.txt', sep='\t', index=False, header=False)
print(final_edgelist_symbols1.head())


