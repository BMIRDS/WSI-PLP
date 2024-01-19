library(TCGAbiolinks)
library(plyr)
library(maftools)
library(dplyr)

maf <- GDCquery_Maf("LGG", pipelines = "muse") %>% read.maf
df = data.frame(maf@data)

table(df$Hugo_Symbol)
sum(df$Hugo_Symbol == 'IDH1')
sum(df$Hugo_Symbol == 'IDH2')

mut_IDH = ddply(df,.(Tumor_Sample_Barcode),function(x) c(max(x$Hugo_Symbol == 'IDH1'),max(x$Hugo_Symbol == 'IDH2')))
names(mut_IDH) <- c("Tumor_Sample_Barcode", "IDH1", "IDH2")
table(mut_IDH$IDH1,mut_IDH$IDH2)

mut_IDH$Tumor_Sample_Barcode <- as.character(mut_IDH$Tumor_Sample_Barcode)
mut_IDH$submitter_id <- unlist(lapply(mut_IDH$Tumor_Sample_Barcode,substr,start=1,stop=12))

mut_IDH$idh = as.integer(apply(mut_IDH, 1, function(x) max(x['IDH1'],x['IDH2'])))

# merge with LGG grade 2
df_dict = read.csv('data/meta_clinical_LGG.csv')
df_mut <- merge(mut_IDH,df_dict,by='submitter_id',all.x=F,all.y=T)
summary(df_mut)
write.csv(df_mut,file='data/meta_clinical_LGG.csv')