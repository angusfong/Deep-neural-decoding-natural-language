setwd('../Wehbe')

load('fullTR.RData')

types <- c("chars","verbs","motion","emotion","pos","dependency_role","speech")
chars <- c("draco","filch","harry","herm","hooch","minerva","neville","peeves","ron","wood")
verbs <- c("be","hear","know","see","tell")
#speech <- c("speak_sticky","speak")
speech <- c("speak")
#motion <- c("fly_sticky","manipulate_sticky","move_sticky","collidePhys_sticky","fly","manipulate","move")
motion <- c("fly","manipulate","move") #these are the persistent ones
#emotion <- c("annoyed","commanding","dislike","fear","like","nervousness","questioning","wonder","annoyed_sticky",
#	 "commanding_sticky","cynical_sticky","dislike_sticky","fear_sticky","hurtMental_sticky","hurtPhys_sticky",
#	 "like_sticky","nervousness_sticky","pleading_sticky","praising_sticky","pride_sticky","questioning_sticky",
#	 "relief_sticky","wonder_sticky")
emotion <- c("annoyed_sticky","commanding_sticky","cynical_sticky","dislike_sticky","fear_sticky","hurtMental_sticky","hurtPhys_sticky",
"like_sticky","nervousness_sticky","pleading_sticky","praising_sticky","pride_sticky","questioning_sticky",
"relief_sticky","wonder_sticky")
visual <- c("word_length","var_WL","sentence_length")
pos <- c(",",".",":","CC","CD","DT","IN","JJ","MD","NN","NNP","NNS","POS","PRP","PRP$","RB","RP","TO","UH","VB",
    "VBD","VBG","VBN","VBP","VBZ","WDT","WP","WRB")
dependency_role <- c("ADV","AMOD","CC","COORD","DEP","IOBJ","NMOD","OBJ","P","PMOD",
		"PRD","PRN","PRT","ROOT","SBJ","VC","VMOD")

interested <- c(chars, verbs,speech,motion,emotion,visual,pos,dependency_role)

fullTR_interested <- fullTR[,interested]

feature_sparsity <- apply(fullTR_interested, 2, function(x) sum(x>0))

save(feature_sparsity, file="feature_sparsity.RData")

