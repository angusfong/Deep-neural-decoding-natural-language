#run visualize_features and get_words before this
#library(caret)
library(reticulate)
np <- import("numpy")
library(e1071)
library(ROCR)
library("mltools")

args = commandArgs(trailingOnly=TRUE)
var_ind <- as.numeric(args[1])

layer <- args[2] # '_memory' or '_embed_final'

empty_lines <- c(1:10,337:350,689:702,968:981,1348:1351)

cat("Variable", var_ind, "\n")

lengths <- c("0","1", "4","8","16","100","200","400","800","all")

load("fullTR.RData")

cat(names(fullTR)[var_ind],"\n")

run_svm <- function(x,y,nfold, kernel='linear') {
  set.seed(1)

  dat = cbind.data.frame(x,y)
  names(dat) <- c(paste("node",1:50,sep=""),"y")
  preds <- rep(NA, length(y))
  
  folds = folds(x=dat$y,nfolds=nfold,stratified=TRUE,seed=1)

  cat(table(folds), "\n")

  cat(sapply(1:nfold, function (i) sum(dat$y[folds==i])), "\n")

  for (i in 1:nfold) {
    train <- dat[folds!=i,]
    test <- dat[folds==i,]
    m_harry <- svm(factor(y) ~ ., data=train,
                   gamma=1, kernel = kernel, cost = 1)
    preds[folds==i] <- as.logical(predict(m_harry, newdata=test))
  }
    
  #ROC curve analysis
  pred_object <- prediction(as.numeric(preds), y)
  auc <- slot(performance(pred_object, "auc"),"y.values")[[1]]
  acc <- max(slot(performance(pred_object, "acc"), "y.values")[[1]],na.rm=TRUE)
  f <- max(slot(performance(pred_object, "f"), "y.values")[[1]],na.rm=TRUE)
  
  print("Confusion matrix:") 
  cm <- table(preds, y)
  print(cm)
  print(paste("Accuracy =", acc))
  print(paste("AUC =", auc))
  print(paste("f =", f))
  
  return(list('preds'=preds, 'cm'=cm, 'acc'=acc, 'auc'=auc, 'f'=f))

}

#The following is how the layer 1 embeddings across sentences were bounded into an array for each context,
#and further combnied into context_embeddings.RData
#ptm <- proc.time()
#for (context in contexts) {
#  layer1 <- NULL
#  for (s in 1:398) {
#    layer1 <- rbind(layer1,np$load(paste("sentence",s,"_layer1_", context, "embeddings.npy",sep="")))
#  }
#  assign(paste("layer1_", context, sep=""), layer1)
#}
#proc.time() - ptm
#save("layer1_","layer1_1w_","layer1_2w_","layer1_4w_","layer1_8w_","layer1_16w_","layer1_32w_","layer1_1s_",
#  "layer1_2s_","layer1_5s_","layer1_docwise_",file="context_embeddings.RData")

aucs <- list()
cms <- list()
accs <- list()
fs <- list()

for (length in lengths) {

  cat("Context:", length,"words \n")

  embeddings = readLines(paste("context-",length,layer,sep=""))

  embeddings_num <- t(sapply(embeddings,function(s) as.numeric((strsplit(s, ' ')[[1]]))))
  embeddings_noempty <- embeddings_num[-empty_lines,]

  set.seed(1)
  shuffle_inds <- sample(1:nrow(embeddings_noempty))
  embeddings_noempty <- embeddings_noempty[shuffle_inds,]
  var <- fullTR[shuffle_inds,var_ind]

  #binarize var, whatever it is
  var <- var > 0  

  #run svm prediction
  preds <- run_svm(embeddings_noempty,var,20)

  aucs[length] <- preds$auc
  accs[length] <- preds$auc
  fs[length] <- preds$f
  cms[length] <- preds$cm

}

varname <- names(fullTR)[var_ind]
save(aucs,file=paste(varname, "_aucs_peng",layer,".RData", sep=""))