from collections import namedtuple

LabelStruct = namedtuple("LabelStruct", "fileName totalCount maxCount avgCount resFile initialFeatureProjMatrixFile clusterFile");

labelStruct0 = LabelStruct(fileName="Toy_Example/bibtex.pkl", totalCount=159, maxCount=17, avgCount=2.4035, resFile='res_bibtex', initialFeatureProjMatrixFile='', clusterFile = '');

#3993
labelStruct1 = LabelStruct(fileName="../DataSets/Eurlex/eurlex.pkl", totalCount=3786, maxCount=24, avgCount=5.311153, resFile='res_eurlex', initialFeatureProjMatrixFile='WeightMatrices/eurlex_OvR_weights.pkl', clusterFile = '');

labelStruct2 = LabelStruct(fileName="../DataSets/Wiki10/wiki10.pkl", totalCount=30938, maxCount=30, avgCount=18.641665, resFile='res_wiki10', initialFeatureProjMatrixFile='', clusterFile = '');

labelStruct3 = LabelStruct(fileName="../DataSets/AmazonCat/amazonCat.pkl", totalCount=13330, maxCount=57, avgCount=5.040670, resFile='res_amazonCat', initialFeatureProjMatrixFile='', clusterFile = 'ClusterCenters/res_amazonCat_centers_C0.1.pkl');

# For delicious large, maxCount is 13203, which is very large. Using avgCount instead
labelStruct4 = LabelStruct(fileName="../DataSets/DeliciousLarge/deliciousLarge.pkl", totalCount=205443, maxCount=75.544836, avgCount=75.544836, resFile='res_deliciousLarge', initialFeatureProjMatrixFile='', clusterFile = 'ClusterCenters/res_deliciousLarge_centers_C0.01.pkl');

# For ALOI, the maxCount is 1, which is very small. Using 10 instead
labelStruct5 = LabelStruct(fileName="../DataSets/ALOI/aloi_scale.pkl", totalCount=1000, maxCount=10, avgCount=1, resFile='res_aloi', initialFeatureProjMatrixFile='WeightMatrices/aloi_OvR_weights.pkl', clusterFile = '');

# For wikiLSHTC, the maxCount is 198. Using 10 instead
labelStruct6 = LabelStruct(fileName="../DataSets/WikiLSHTC/wikiLSHTC.pkl", totalCount=325056, maxCount=10, avgCount=3.192231, resFile='res_wikiLSHTC', initialFeatureProjMatrixFile='', clusterFile = 'ClusterCenters/res_wikiLSHTC_centers_C1.pkl');

# Taking arbitrary value for maxCount and avgCount
labelStruct7 = LabelStruct(fileName="../DataSets/RelatedSearch/related_search_Xdense.pkl", totalCount=1944958, maxCount=10, avgCount=1.592231, resFile='res_relatedSearch', initialFeatureProjMatrixFile='', clusterFile = '');

# maxCount and avgCount are unset
labelStruct8 = LabelStruct(fileName="../DataSets/RelatedSearch/related_search_Xdense_L5K.pkl", totalCount=5000, maxCount=0, avgCount=0, resFile='res_relatedSearch_L5K', initialFeatureProjMatrixFile='', clusterFile = '');

# maxCount and avgCount are unset
labelStruct9 = LabelStruct(fileName="../DataSets/RelatedSearch/related_search_Xdense_L10K.pkl", totalCount=10000, maxCount=0, avgCount=0, resFile='res_relatedSearch_L10K', initialFeatureProjMatrixFile='', clusterFile = '');

# maxCount and avgCount are unset
labelStruct10 = LabelStruct(fileName="../DataSets/RelatedSearch/related_search_Xdense_L100K.pkl", totalCount=100000, maxCount=0, avgCount=0, resFile='res_relatedSearch_L100K', initialFeatureProjMatrixFile='', clusterFile = '');

# maxCount and avgCount are unset
labelStruct11 = LabelStruct(fileName="../DataSets/RelatedSearch/related_search_X2dense_L5K.pkl", totalCount=5000, maxCount=0, avgCount=0, resFile='res_relatedSearch_X2_L5K', initialFeatureProjMatrixFile='', clusterFile = '');

# maxCount and avgCount are unset
labelStruct12 = LabelStruct(fileName="../DataSets/RelatedSearch/related_search_X2dense_L10K.pkl", totalCount=10000, maxCount=0, avgCount=0, resFile='res_relatedSearch_X2_L10K', initialFeatureProjMatrixFile='', clusterFile = '');

# maxCount and avgCount are unset
labelStruct13 = LabelStruct(fileName="../DataSets/RelatedSearch/related_search_Xdense_L5K_S1.pkl", totalCount=5000, maxCount=0, avgCount=0, resFile='res_relatedSearch_L5K_S1', initialFeatureProjMatrixFile='', clusterFile = '');

labelStruct14 = LabelStruct(fileName="../DataSets/Delicious/delicious.pkl", totalCount=983, maxCount=0, avgCount=0, resFile='res_delicious', initialFeatureProjMatrixFile='WeightMatrices/delicious_OvR_weights.pkl', clusterFile = '');

labelStruct15 = LabelStruct(fileName="../DataSets/DeliciousLarge/deliciousLarge_1k.pkl", totalCount=1000, maxCount=0, avgCount=0, resFile='res_deliciousLarge_1k', initialFeatureProjMatrixFile='', clusterFile = '');

labelStruct16 = LabelStruct(fileName="../DataSets/DeliciousLarge/deliciousLarge_5k.pkl", totalCount=5000, maxCount=0, avgCount=0, resFile='res_deliciousLarge_5k', initialFeatureProjMatrixFile='', clusterFile = '');

labelStruct17 = LabelStruct(fileName="../DataSets/DeliciousLarge/deliciousLarge_2k.pkl", totalCount=2000, maxCount=0, avgCount=0, resFile='res_deliciousLarge_2k', initialFeatureProjMatrixFile='', clusterFile = '');

labelStruct18 = LabelStruct(fileName="../DataSets/Amazon/amazon.pkl", totalCount=0, maxCount=0, avgCount=0, resFile='res_amazon', initialFeatureProjMatrixFile='', clusterFile = '');

labelStruct19 = LabelStruct(fileName="../DataSets/WikiLSHTC/wikiLSHTC_100K.pkl", totalCount=0, maxCount=0, avgCount=0, resFile='res_wikiLSHTC_100K', initialFeatureProjMatrixFile='', clusterFile = '');

labelStruct20 = LabelStruct(fileName="../DataSets/DeliciousLarge/deliciousLarge_50K.pkl", totalCount=0, maxCount=0, avgCount=0, resFile='res_deliciousLarge_50K', initialFeatureProjMatrixFile='', clusterFile = '');

labelStruct21 = LabelStruct(fileName="../DataSets/Odp/odp_500K.pkl", totalCount=0, maxCount=0, avgCount=0, resFile='res_odp_500K', initialFeatureProjMatrixFile='', clusterFile = '');

labelStruct22 = LabelStruct(fileName="../DataSets/Odp/odp.pkl", totalCount=0, maxCount=0, avgCount=0, resFile='res_odp', initialFeatureProjMatrixFile='', clusterFile = '');

labelStructs = [labelStruct0, labelStruct1, labelStruct2, labelStruct3, labelStruct4, labelStruct5, labelStruct6, labelStruct7, labelStruct8, labelStruct9, labelStruct10, labelStruct11, labelStruct12, labelStruct13, labelStruct14, labelStruct15, labelStruct16, labelStruct17, labelStruct18, labelStruct19, labelStruct20, labelStruct21, labelStruct22];
