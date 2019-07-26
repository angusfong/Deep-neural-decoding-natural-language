setwd('../Wehbe')

load('fullTR.RData')
original <- fullTR
contexts = c(0,1,2,4,16,1600)

for (context in contexts) {
	fullTR <- original
	for (i in 1:nrow(fullTR)) {
		first_row <- max(0, i-context)
		fullTR[i,] <- colSums(original[first_row:i,]) 
	}
	save("fullTR", file=paste('fullTR_', context, 's_.RData',sep=''))
}
