library(gridGraphics)
library(grid)
library(gplots)
library(colorspace)
library(RColorBrewer)

#opt
dec <- "wb_dec"
opt <- "TR" #"TR_", or "sentence_"
layer <- "layer1"
sub = '1'

#glove
glove = FALSE
if glove {
	setwd() = '/gpfs/milgram/project/chun/kxt3/official/glove/aucs'
}

types <- c("chars","verbs","motion","emotion","pos","dependency_role","speech")

chars <- c("draco","filch","harry","herm","hooch","minerva","neville","peeves","ron","wood")
verbs <- c("be","hear","know","see","tell")
#speech <- c("speak_sticky","speak")
speech <- c("speak")
#motion <- c("fly_sticky","manipulate_sticky","move_sticky","collidePhys_sticky","fly","manipulate","move")
motion <- c("fly","manipulate","move")
#emotion <- c("annoyed","commanding","dislike","fear","like","nervousness","questioning","wonder","annoyed_sticky",
#	"commanding_sticky","cynical_sticky","dislike_sticky","fear_sticky","hurtMental_sticky","hurtPhys_sticky",
#	"like_sticky","nervousness_sticky","pleading_sticky","praising_sticky","pride_sticky","questioning_sticky",
#	"relief_sticky","wonder_sticky")
emotion <- c("annoyed_sticky","commanding_sticky","cynical_sticky","dislike_sticky","fear_sticky","hurtMental_sticky","hurtPhys_sticky",
"like_sticky","nervousness_sticky","pleading_sticky","praising_sticky","pride_sticky","questioning_sticky",
"relief_sticky","wonder_sticky")
visual <- c("word_length","var_WL","sentence_length")
pos <- c(",",".",":","CC","CD","DT","IN","JJ","MD","NN","NNP","NNS","POS","PRP","PRP$","RB","RP","TO","UH","VB",
	"VBD","VBG","VBN","VBP","VBZ","WDT","WP","WRB")
dependency_role <- c("ADV","AMOD","CC","COORD","DEP","IOBJ","NMOD","OBJ","P","PMOD",
	"PRD","PRN","PRT","ROOT","SBJ","VC","VMOD")

# decoded
if (TRUE) {
	decoded = NULL
	for (type in c("chars","verbs","motion","emotion","pos","dependency_role","speech")) {
		x = NULL
		for (varname in get(type)) {
			load(paste("subject", sub, "_wb_dec_", opt, "_", layer ,"_", varname, "_aucs.RData", sep=""))
			x = cbind(x, unlist(aucs))

		}
		x = t(x)
		x = x[ order(rowMeans(x), decreasing=T), ]
		rownames = get(type)
		if (is.null(dim(x))) {
			x = rbind(x,x)
			rownames = rep(rownames,2)
		}
		row.names(x) = rownames
		colnames(x) = c('0','1','2','4','16','doc')
		assign(paste(type, '_aucs_decoded', sep=''), x)
		decoded = rbind(decoded, x)
	}
}

#actual
actual = NULL
for (type in types) {
	x = NULL
	for (varname in get(type)) {
		load(paste("subject", sub, "__", opt, "_", layer ,"_", varname, "_aucs.RData", sep=""))
		x = cbind(x, unlist(aucs))

	}
	x = t(x)
	x = x[ order(rowMeans(x), decreasing=T), ]
	rownames = get(type)
	if (is.null(dim(x))) {
		x = rbind(x,x)
		rownames = rep(rownames,2)
	}
	row.names(x) = rownames
	colnames(x) = c('0','1','2','4','16','doc')
	assign(paste(type, '_aucs_actual', sep=''), x)
	actual = rbind(actual, x)
}

grab_grob <- function(){
	grid.echo()
	grid.grab()
}


#rowmeans: apply(emotion_aucs, 1, function(x) mean(as.numeric(x)))
varnames = c(paste(types, '_aucs_actual', sep=''), paste(types, '_aucs_decoded', sep=''))

setwd('results')

library(gplots)
arr <- replicate(4, matrix(sample(1:100),nrow=10,ncol=10), simplify = FALSE)

if (FALSE) {
	gl <- lapply(1:4, function(i){
		heatmap.2(arr[[i]],Rowv = NA, Colv = NA,trace = 'none', breaks=seq(1,101), col=diverge_hcl(100),offsetRow = -36)   
		grab_grob()
	})
}

if (TRUE) {
	gl <- sapply(varnames, function(varname){
		pdf(varname)
		heatmap.2(get(varname),Rowv = NA, Colv = NA,trace = 'none', breaks=seq(0.5,1,0.01), main=varname,col=diverge_hcl(50),offsetRow = -36)
		dev.off()
		#grab_grob()
	})
}

grid.newpage()
library(gridExtra)
grid.arrange(grobs=gl, ncol=2, clip=TRUE)
