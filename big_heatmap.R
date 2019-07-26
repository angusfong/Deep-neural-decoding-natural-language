setwd('../Wehbe')

library(gridGraphics)
library(grid)
library(gplots)
library(colorspace)
library(RColorBrewer)

include_longs <- FALSE

subs <- c('1','2','3','4','5','6','7','8')
thres <- 0.05

types <- c("chars","verbs","motion","emotion","speech")
chars <- c("draco","filch","harry","herm","hooch","minerva","neville","peeves","ron","wood")
verbs <- c("be","hear","know","see","tell")
speech_punctual <- c("speak_sticky"); speech_sticky <- c("speak")
speech <- c(speech_punctual,speech_sticky)
motion_punctual <- c("fly_sticky","manipulate_sticky","move_sticky")
motion_sticky <- c("fly","manipulate","move","collidePhys_sticky") #these are the persistent ones
motion <- c(motion_punctual, motion_sticky)
emotion_punctual <- c("annoyed","commanding","dislike","fear","like","nervousness","questioning","wonder")
emotion_sticky <- c("annoyed_sticky","commanding_sticky","cynical_sticky","dislike_sticky","fear_sticky","hurtMental_sticky","hurtPhys_sticky",
"like_sticky","nervousness_sticky","pleading_sticky","praising_sticky","pride_sticky","questioning_sticky",
"relief_sticky","wonder_sticky")
emotion <- c(emotion_punctual,emotion_sticky)
visual <- c("word_length","var_WL","sentence_length")
pos <- c(",",".",":","CC","CD","DT","IN","JJ","MD","NN","NNP","NNS","POS","PRP","PRP$","RB","RP","TO","UH","VB",
	"VBD","VBG","VBN","VBP","VBZ","WDT","WP","WRB")
dependency_role <- c("ADV","AMOD","CC","COORD","DEP","IOBJ","NMOD","OBJ","P","PMOD",
	"PRD","PRN","PRT","ROOT","SBJ","VC","VMOD")

# correct
correct <- function(s) {
	r <- s
	if (s %in% c("fly_sticky","manipulate_sticky","move_sticky","speak_sticky")) r <- substr(r,1,nchar(r)-7)
	if (s %in% c("fly","manipulate","move","speak")) r <- paste(r,"_sticky",sep="")
	return(r)
}

#eval against a null dist
minusone_if_insig <- function(stat_list, null_list, model) {
	return(sapply(1:length(stat_list), 
		function(con) {
			if (mean(unlist(stat_list[[con]]) < unlist(null_list[[con]])) < thres) return(unlist(stat_list[[con]])) else return(-1)
		}))
}

#actual lstm
lstm_actual_big = NULL
for (type in types) {
	x = NULL
	for (varname in get(type)) {
		#filename: subject1__TR_layer1_harry_aucs.RData
		load(paste("subject1__TR_layer1_", varname, "_aucs.RData", sep=""))
		x = cbind(x, minusone_if_insig(pr_aucs, pr_perm_aucs, "lstm"))

	}
	x = t(x)
	rownames = paste(type, '_', sapply(get(type),function(s) correct(s)),sep='')
	if (is.null(dim(x))) {
		x = rbind(x,x)
		rownames = rep(rownames,2)
	}
	row.names(x) = rownames
	x = x[ order(rowMeans(x), decreasing=T), ]
	colnames(x) = c('0','1','2','4','16','doc')
	assign(paste(type, '_aucs_actual', sep=''), x)
	lstm_actual_big = rbind(lstm_actual_big, x)
}


#actual glove
glove_actual_big = NULL
for (type in types) {
	x = NULL
	for (varname in get(type)) {
		#filename: subject1__TR_glove_harry_aucs.RData
		load(paste("subject1__TR_glove_", varname, "_aucs.RData", sep=""))
		x = cbind(x, minusone_if_insig(pr_aucs, pr_perm_aucs, "glove"))

	}
	x = t(x)
	rownames = paste(type, '_', sapply(get(type),function(s) correct(s)),sep='')
	if (is.null(dim(x))) {
		x = rbind(x,x)
		rownames = rep(rownames,2)
	}
	row.names(x) = rownames
	x = x[ order(rowMeans(x), decreasing=T), ]
	colnames(x) = c('0','1','2','4','16','doc')
	assign(paste(type, '_aucs_actual', sep=''), x)
	glove_actual_big = rbind(glove_actual_big, x)
}



# decoded lstm
lstm_decoded_big = NULL
files <- list.files()
for (type in types) {
	x = NULL
	for (varname in get(type)) {
		print(varname)
		if (paste("subject1_sl5000_TR_layer1_", varname, "_aucs.RData", sep="") %in% files) {
			#filename:subject1_sl5000_TR_layer1_harry_aucs.RData
			mean_pr_aucs <- NULL
			mean_pr_perm_aucs <- NULL
			for (sub in subs) {
				load(paste("subject",sub,"_sl5000_TR_layer1_", varname, "_aucs.RData", sep=""))				
				if (is.null(mean_pr_perm_aucs)) {
					mean_pr_aucs <- unlist(pr_aucs)
					mean_pr_perm_aucs <- lapply(pr_perm_aucs, sort)
				} else {
					mean_pr_aucs <- mean_pr_aucs + unlist(pr_aucs)
					pr_perm_aucs <- lapply(pr_perm_aucs, sort)
					mean_pr_perm_aucs <- lapply(names(pr_perm_aucs), function(x) mean_pr_perm_aucs[x][[1]] + pr_perm_aucs[x][[1]])
					names(mean_pr_perm_aucs) <- names(pr_perm_aucs)
				}				
			}
			mean_pr_aucs <- mean_pr_aucs/8
			mean_pr_perm_aucs <- lapply(mean_pr_perm_aucs, function(x) x/8)
			x = cbind(x, minusone_if_insig(mean_pr_aucs, mean_pr_perm_aucs, "lstm"))
		}
	}
	x = t(x)
	rownames = paste(type, '_', sapply(get(type),function(s) correct(s)),sep='')
	if (is.null(dim(x))) {
		x = rbind(x,x)
		rownames = rep(rownames,2)
	}
	row.names(x) = rownames
	x = x[ order(rowMeans(x), decreasing=T), ]
	colnames(x) = c('0','1','2','4','16','doc')
	assign(paste(type, '_aucs_decoded', sep=''), x)
	lstm_decoded_big = rbind(lstm_decoded_big, x)
}



# decoded glove
glove_decoded_big = NULL
files <- list.files()
for (type in types) {
	x = NULL
	for (varname in get(type)) {
		print(varname)
		if (paste("subject1_sl5000_TR_glove_", varname, "_aucs.RData", sep="") %in% files) {
			#filename:subject1_sl5000_TR_glove_harry_aucs.RData
			mean_pr_aucs <- NULL
			mean_pr_perm_aucs <- NULL
			for (sub in subs) {
				load(paste("subject",sub,"_sl5000_TR_glove_", varname, "_aucs.RData", sep=""))				
				if (is.null(mean_pr_perm_aucs)) {
					mean_pr_aucs <- unlist(pr_aucs)
					mean_pr_perm_aucs <- lapply(pr_perm_aucs, sort)
				} else {
					mean_pr_aucs <- mean_pr_aucs + unlist(pr_aucs)
					pr_perm_aucs <- lapply(pr_perm_aucs, sort)
					mean_pr_perm_aucs <- lapply(names(pr_perm_aucs), function(x) mean_pr_perm_aucs[x][[1]] + pr_perm_aucs[x][[1]])
					names(mean_pr_perm_aucs) <- names(pr_perm_aucs)
				}				
			}
			mean_pr_aucs <- mean_pr_aucs/8
			mean_pr_perm_aucs <- lapply(mean_pr_perm_aucs, function(x) x/8)
			x = cbind(x, minusone_if_insig(mean_pr_aucs, mean_pr_perm_aucs, "glove"))
		}
	}
	x = t(x)
	rownames = paste(type, '_', sapply(get(type),function(s) correct(s)),sep='')
	if (is.null(dim(x))) {
		x = rbind(x,x)
		rownames = rep(rownames,2)
	}
	row.names(x) = rownames
	x = x[ order(rowMeans(x), decreasing=T), ]
	colnames(x) = c('0','1','2','4','16','doc')
	assign(paste(type, '_aucs_decoded', sep=''), x)
	glove_decoded_big = rbind(glove_decoded_big, x)
}

setwd('/gpfs/milgram/project/chun/hf246/Language/Wehbe/results')

lstm_actual_file = paste('lstm_actual_big.pdf',sep='')
lstm_actual_shuffled_file = paste('lstm_actual_shuffled_big.pdf',sep='')
lstm_actual_long_file = paste('lstm_actual_long_big.pdf',sep='')
lstm_decoded_file = paste('lstm_decoded_big.pdf', sep='')
lstm_decoded_long_file = paste('lstm_decoded_long_big.pdf',sep='')
glove_actual_file = paste('glove_actual_big.pdf',sep='')
glove_actual_shuffled_file = paste('glove_actual_shuffled_big.pdf',sep='')
glove_actual_long_file = paste('glove_actual_long_big.pdf',sep='')
glove_decoded_file = paste('glove_decoded_big.pdf', sep='')
glove_decoded_long_file = paste('glove_decoded_long_big.pdf',sep='')
blues <- brewer.pal(100,"Blues")
col_scale=blues[c(1:2,4:9)]
palette = colorRampPalette(col_scale, space="rgb", bias=1)

pdf(lstm_actual_file, width=8, height=10)
heatmap.2(lstm_actual_big,density.info='none',Rowv = NA, Colv = NA,trace = 'none', breaks=seq(0,1,0.01), 
	main=paste("Actual LSTM", sep=''),col=palette(100),offsetRow = -40,cellnote=round(lstm_actual_big,2), notecex=.7,
	notecol="black",na.color=par("bg"),lhei=c(1,8),srtCol=0)
dev.off()

pdf(glove_actual_file, width=8, height=10)
heatmap.2(glove_actual_big,density.info='none',Rowv = NA, Colv = NA,trace = 'none', breaks=seq(0,1,0.01), 
	main=paste("Actual GloVe", sep=''),col=palette(100),offsetRow = -40,cellnote=round(glove_actual_big,2), notecex=.7,
	notecol="black",na.color=par("bg"),lhei=c(1,8),srtCol=0)
dev.off()

pdf(lstm_decoded_file, width=8, height=10)
heatmap.2(lstm_decoded_big,density.info='none',Rowv = NA, Colv = NA,trace = 'none', breaks=seq(0,1,0.01), 
	main=paste("Decoded LSTM - Averaged results", sep=''),col=palette(100),offsetRow = -40,cellnote=round(lstm_decoded_big,2), notecex=.7,
	notecol="black",na.color=par("bg"),lhei=c(1,8),srtCol=0)
dev.off()

pdf(glove_decoded_file, width=8, height=10)
heatmap.2(glove_decoded_big,density.info='none',Rowv = NA, Colv = NA,trace = 'none', breaks=seq(0,1,0.01), 
	main=paste("Decoded GloVe - Averaged results", sep=''),col=palette(100),offsetRow = -40,cellnote=round(glove_decoded_big,2), notecex=.7,
	notecol="black",na.color=par("bg"),lhei=c(1,8),srtCol=0)
dev.off()

