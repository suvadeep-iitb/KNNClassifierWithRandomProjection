from collections import namedtuple

LabelStruct = namedtuple("LabelStruct", "fileName totalCount maxCount avgCount resFile");

labelStruct0 = LabelStruct(fileName="Toy_Example/bibtex.pkl", totalCount=159, maxCount=17, avgCount=2.4035, resFile='res_bibtex');

labelStruct1 = LabelStruct(fileName="../DataSets/Eurlex/eurlex.pkl", totalCount=3993, maxCount=24, avgCount=5.311153, resFile='res_eurlex');

labelStruct2 = LabelStruct(fileName="../DataSets/Wiki10/wiki10.pkl", totalCount=30938, maxCount=30, avgCount=18.641665, resFile='res_wiki10');

labelStruct3 = LabelStruct(fileName="../DataSets/AmazonCat/amazonCat.pkl", totalCount=13330, maxCount=57, avgCount=5.040670, resFile='res_amazonCat');

# For delicious large, maxCount is 13203, which is very large. Using avgCount instead
labelStruct4 = LabelStruct(fileName="../DataSets/DeliciousLarge/deliciousLarge.pkl", totalCount=205443, maxCount=75.544836, avgCount=75.544836, resFile='res_deliciousLarge');

# For ALOI, the maxCount is 1, which is very small. Using 10 instead
labelStruct5 = LabelStruct(fileName="../DataSets/ALOI/aloi_scale.pkl", totalCount=1000, maxCount=10, avgCount=1, resFile='res_aloi');

# For wikiLSHTC, the maxCount is 198. Using 10 instead
labelStruct6 = LabelStruct(fileName="../DataSets/WikiLSHTC/wikiLSHTC.pkl", totalCount=325056, maxCount=10, avgCount=3.192231, resFile='res_wikiLSHTC');


labelStructs = [labelStruct0, labelStruct1, labelStruct2, labelStruct3, labelStruct4, labelStruct5, labelStruct6];
