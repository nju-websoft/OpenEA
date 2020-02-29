import openea.modules.finding.alignment as search

import openea.modules.load.read as dfunc

training_folder = "../../../dataset_20180928/_1/dbp_wd_15k_V1_1/"
output_folder = "../../../dataset_20180928/out/BootEA/dbp_wd_15k_V1_1/20181110165244/"
metric = 'inner'
div = '631/'
normalize = False
csls_k = 10
accurate = True
threads_num = 8
top_k = [1, 5, 10, 50]

ent_embeds = dfunc.load_embeddings(output_folder + "ent_embeds.npy")
assert ent_embeds is not None
mapping_mat = dfunc.load_embeddings(output_folder + "mapping_mat.npy")
test_links = dfunc.read_links(training_folder + div + 'test_links')
ent_ids1 = dfunc.read_dict(output_folder + 'kg1_ent_ids')
ent_ids2 = dfunc.read_dict(output_folder + 'kg2_ent_ids')
test_ids1 = [ent_ids1[link[0]] for link in test_links]
test_ids2 = [ent_ids2[link[1]] for link in test_links]
embed1 = ent_embeds[test_ids1, ]
embed2 = ent_embeds[test_ids2, ]
print("re-retrieval", output_folder)

search.stable_alignment(embed1, embed2, 'cosine', False, csls_k=csls_k)

#
# search.test(embed1, embed2, mapping_mat, top_k, threads_num,
#             metric='cosine', normalize=False, csls_k=0, accurate=accurate)
#
#
# search.test(embed1, embed2, mapping_mat, top_k, threads_num,
#             metric='cosine', normalize=False, csls_k=5, accurate=accurate)
#
# search.test(embed1, embed2, mapping_mat, top_k, threads_num,
#             metric='cosine', normalize=False, csls_k=10, accurate=accurate)
