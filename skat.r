library('SKAT')

data("SKAT.haplotypes")
attach(SKAT.haplotypes)
hp = Haplotype
# dim(hp)

write.csv(hp, './hp.csv', quote = FALSE)
print('SKAT.haplotypes written in ./hp.csv')