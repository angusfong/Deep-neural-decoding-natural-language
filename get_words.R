setwd("/Users/angusfong/Documents/Yale/Academic/Chun_lab/Decoding_project/Wehbe")
library(R.matlab)
m = readMat("story_features.mat")$features
#length(m[,,1]$type)
#m[,,1]$type[[1]]

#length(m[,,6]$names)
#m[,,6]$names

#dim(m[,,1]$values) #nword by nfeature
#m[,,1]$values

for (i in 1:10) {
  f <- m[,,i]
  nam <- f$type
  assign(nam, as.data.frame(matrix(f$values, nrow=5176, dimnames=list(NULL, unlist(f$names)))))
}
gset
full = cbind(NNSE, Speech, Motion, Emotion, Verbs, Characters, Visual,
             Word_Num,Part_of_speech,Dependency_role)

setnames = c("NNSE", "Speech", "Motion", "Emotion", "Verbs", 
             "Characters", "Visual", "Word_Num","Part_of_speech",
             "Dependency_role")

pdf("story_features_visual.pdf", width=12, height=5)
xticks = seq(4,5176,4)
for (setname in setnames) {
  set = get(setname)
  for (var in 1:ncol(set)) {
    sparsity = mean(set[,var]!=0)
    plot(set[,var],xlab="word",type="lines",main=paste(setname,"_",names(set)[var]," sparsity = ", round(sparsity,2),sep=""))
    #axis(side=1, at=xticks, labels = FALSE)
  }
}
dev.off()

pdf("story_features_visual_TR.pdf", width=12, height=5)
xticks = seq(1,1294)
for (setname in setnames) {
  set = get(setname)
  setTR = data.frame(matrix(NA,1294,length(names(set))))
  names(setTR) = names(set)
  for (segment in 1:1294) {
    if (ncol(setTR) > 1) {
      setTR[segment,] = colSums(set[(segment*4-3):(segment*4),])
    } else {
      setTR[segment,] = sum(set[(segment*4-3):(segment*4),])
    }
  }
  nam = paste(setname, "TR", sep="")
  assign(nam, setTR)
  for (var in 1:ncol(setTR)) {
    sparsity = mean(setTR[,var]!=0)
    plot(setTR[,var],xlab="TR",type="lines",main=paste(setname,"_",names(setTR)[var]," sparsity = ", round(sparsity,2),sep=""))
    #axis(side=1, at=xticks, labels = FALSE)
  }
}
dev.off()

# are some categorical vars mutually exclusive?
fullTR = data.frame(matrix(NA,1294,length(names(full))))
names(fullTR) = names(full)
for (segment in 1:1294) {
  fullTR[segment,] = colSums(full[(segment*4-3):(segment*4),])
}

summary(full$annoyed_sticky)
