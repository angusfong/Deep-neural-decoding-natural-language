# Set up parameters
args = commandArgs(trailingOnly=TRUE)
var_ind <- as.numeric(args[1]); layer <- args[2]; sub <- args[3]
#var_ind <- 113; layer <- 'layer1'; sub <- '1'
# use actual or decoded?
dec = "sl5000" # "" or "wb" or "sl5000"
# predict features in the entire context window or just the most recent TR?
long = "" # "" or "_long" 

# use sentence or TR
opt = "TR" #sentence or TR
# shuffle labels? (to get a null results)
shuffle <- "" # "" or "_shuffled"
# for perm tests
nperm <- 1000

setwd('../Wehbe')

# define output path
output_path <- "/gpfs/milgram/project/chun/hf246/Language/Wehbe/"

#run visualize_features and get_words before this
#library(caret)
library(reticulate)
np <- import("numpy")
library(e1071)
library(ROCR)
library(PRROC)
library("mltools")

# configure number of nodes
if (layer == "layer0" | layer == "layer1") {
  path_to_embeddings <- "/gpfs/milgram/project/chun/hf246/Language/Wehbe/"
  nnodes = 1024
} else if (layer == "glove") {
  path_to_embeddings <- "/gpfs/milgram/project/chun/kxt3/official/glove"
  nnodes = 300
}

# define possible context lengths
if (opt == "TR") {
  nlines <- 1295
  if (long == "_long") {
    lengths <- c("0s_","1s_", "2s_","4s_")  
  } else {
    lengths <- c("0s_","1s_", "2s_","4s_","16s_","1600s_")  
  }
  
} else {
  lengths <- c("", "1w_", "2w_", "4w_", "8w_", "16w_", "1s_", "2s_","5s_", "docwise_"); nlines <- 398
}

run_svm <- function(xtrain, xtest,y,nfold, kernel) {
  set.seed(1)

  dat_train = cbind.data.frame(xtrain,y)
  dat_test = cbind.data.frame(xtest,y)
  names(dat_train) <- c(paste("node",1:nnodes,sep=""),"y")
  names(dat_test) <- c(paste("node",1:nnodes,sep=""),"y")
  preds <- rep(NA, length(y))
  
  folds = folds(x=dat_train$y,nfolds=nfold,stratified=TRUE,seed=1)

  cat(table(folds), "\n")

  cat(sapply(1:nfold, function (i) sum(dat_train$y[folds==i])), "\n")

  for (i in 1:nfold) {
    train <- dat_train[folds!=i,]
    test <- dat_test[folds==i,]
    m <- svm(factor(y) ~ ., data=train, probability=TRUE,
                   gamma=1, kernel = kernel, cost = 1)
    tmp <- predict(m, newdata=test, probability=TRUE)
    #preds[folds==i] <- tmp
    preds[folds==i] <- attr(tmp,'probabilities')[,'TRUE']
  }
  
  #PR curve analysis
  positive_scores <- preds[y==TRUE]
  negative_scores <- preds[y==FALSE]
  pr<-pr.curve(scores.class0 = positive_scores, scores.class1 = negative_scores)
  pr_auc <- pr$auc.integral
  print(paste("PR AUC:", pr_auc))

  #ROC curve analysis
  roc<-roc.curve(scores.class0 = positive_scores, scores.class1 = negative_scores)
  roc_auc <- roc$auc
  print(paste("ROC AUC:", roc_auc))

  #ROC curve analysis
  #pred_object <- prediction(as.numeric(preds), y)
  #auc <- slot(performance(pred_object, "auc"),"y.values")[[1]]
  #acc <- max(slot(performance(pred_object, "acc"), "y.values")[[1]],na.rm=TRUE)
  #f <- max(slot(performance(pred_object, "f"), "y.values")[[1]],na.rm=TRUE)
  
  #print("Confusion matrix:") 
  #cm <- table(preds, y)
  #print(cm)
  #print(paste("Accuracy =", acc))
  #print(paste("AUC =", auc))
  #print(paste("f =", f))

  pr_perm_auc <- rep(0,nperm); roc_perm_auc <- rep(0, nperm)
  for (iter in 1:nperm) {
    perm_preds <- sample(preds)
    positive_scores_perm <- perm_preds[y==TRUE]
    negative_scores_perm <- perm_preds[y==FALSE]
    pr_perm<-pr.curve(scores.class0 = positive_scores_perm, scores.class1 = negative_scores)
    pr_perm_auc[iter] <- pr_perm$auc.integral
    roc_perm<-roc.curve(scores.class0 = positive_scores_perm, scores.class1 = negative_scores)
    roc_perm_auc[iter] <- roc_perm$auc
  }
  
  return(list('preds'=preds, 'pr_auc'=pr_auc, 'roc_auc'=roc_auc, 'pr_perm_auc'=list(pr_perm_auc), 'roc_perm_auc'=list(roc_perm_auc)))
  #return(list('preds'=preds, 'cm'=cm, 'acc'=acc, 'auc'=auc, 'f'=f))

}

#The following is how the layer 1 embeddings across sentences were bounded into an array for each context,
#and further combined into context_embeddings.RData
#ptm <- proc.time()
#for (context in lengths) {
#
#  tmp <- matrix(NA,nlines,nnodes)
#  for (s in 1:nlines) {
#    tmp[s,] <- np$load(paste(dec, opt,s,"_",layer, "_", context, "embeddings.npy",sep=""))
#  }
#  assign(paste(layer, context, sep="_"), tmp)
#}
#proc.time() - ptm
#save(list=(paste(layer,lengths,sep="_")),file=paste(opt, "_", layer, "_context_embeddings.RData", sep=""))

##

setwd(path_to_embeddings)
train_embeddings_filename <- paste(opt, "_", layer, "_context_embeddings.RData", sep="")
load(train_embeddings_filename)
setwd(output_path)

pr_aucs <- list()
pr_perm_aucs <- list()
roc_aucs <- list()
roc_perm_aucs <- list()

for (length in lengths) {

  if (opt == "TR") {
    if (long == "_long") {
      f = paste("fullTR_", length, ".RData", sep="")
    } else {
      f = "fullTR.RData"
    }
    load(f); Y = fullTR
  } else {
    load("fullSentences_processed.RData"); Y = fullSentences_processed
  }
  cat(names(Y)[var_ind],"\n")

  context <- paste(layer,length,sep="_")

  train_embeddings <- get(context)

  if (dec!="") {
    test_embeddings <- np$load(paste("subject", sub, "_", dec, "_", layer, "_", length, "decoded.npy", sep=""))
    if (sub == "AVG") {
      test_embeddings <- matrix(0, nrow = nlines, ncol = nnodes)
      for (s in c("1","2","3","4","5","6","7","8")) {
        test_embeddings <- test_embeddings + np$load(paste("subject", s, "_", dec, "_", layer, "_", length, "decoded.npy", sep=""))
      }
      test_embeddings <- test_embeddings / 8
    }
    setwd(output_path)
  } else {
    test_embeddings <- get(context)
  }

  cat("Context:", context,"\n")

  set.seed(1)
  shuffle_inds <- sample(1:nrow(train_embeddings))
  train_embeddings <- train_embeddings[shuffle_inds,]
  test_embeddings <- test_embeddings[shuffle_inds,]

  var <- Y[shuffle_inds,var_ind]

  #binarize var, whatever it is
  var <- var > 0  

  # shuffle (for perm test)
  if (shuffle=="_shuffled") {
    var <- sample(var)
  }

  #run svm prediction
  xtrain<-train_embeddings
  xtest<-test_embeddings
  y<-var
  nfold<-20
  kernel<-"linear"
  preds <- run_svm(xtrain,xtest,y,nfold,kernel)

  pr_aucs[context] <- preds$pr_auc
  roc_aucs[context] <- preds$roc_auc
  pr_perm_aucs[context] <- preds$pr_perm_auc
  roc_perm_aucs[context] <- preds$roc_perm_auc

}

varname <- names(Y)[var_ind]

setwd(output_path)
f <- paste("subject", sub, "_", dec, "_", opt, "_", layer ,"_", varname, long, shuffle, "_aucs.RData", sep="")
save(pr_aucs, roc_aucs, pr_perm_aucs, roc_perm_aucs, file=f)