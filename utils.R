# This script collects utility functions for the main and ancilliary analysis to reduce redundancy and amount of code.
# Setup
knitr::opts_chunk$set(fig.align = "center", echo = TRUE)
library(tidyverse) # data handling and plotting
library(plyr) # data handling
library(dplyr) # data handling
library(rpart)# model fitting
library(caret) # automated cross-validation
library(knitr) # kniting the markdown file
library(e1071) # modeling
library(doParallel) # parallel computation
library(ROSE) # upsampling (random over-sampling examples)
library(gbm) # feature importance
library(yardstick) # for balanced accuracy
library(plotROC) # for plotting ROC curves                     
library(pROC) # for ROC confidence intervals
library(predtools) # for calibration plots
library(boot) # for bootstrap confidence intervals

# ggplot theme
ggplot_theme<-theme_bw()

# Get main data
get_main_data<-function(){
  result=list()
  load("data_analyse.RData")
  data <- data_analyse_mod
  # Get set of "unique" IDs
  data_unique <- distinct(data, ID, .keep_all = TRUE) 
  
  #Get IDs for train and test sets 
  tr_prop = 0.75    # proportion of full dataset to use for training
  training_set = ddply(data_unique, .(Victim), function(., seed) { set.seed(seed); .[sample(1:nrow(.), trunc(nrow(.) * tr_prop)), ] }, seed = 42)
  training_IDs <- as.vector(training_set$ID)
  
  #change all boolean variables to factors for ROSE algorithm
  boolean_cols<-setNames(as.logical(lapply(data,is.logical)),colnames(data))
  data[boolean_cols]<-lapply(data[boolean_cols],factor,labels = c("No","Yes"),levels = c(F,T))
  
  # Use vector with training IDs to split full dataset
  data_training <- filter(data, ID %in% training_IDs)
  result$data_test <- filter(data, !(ID%in% training_IDs))
  
  # Subset to variables used in the model (reduce number of variables for upsampling and remove "Victim")
  subset <- c("ID", "Crit", "After_RTP", "Age", "Pos_code", "VV_resid_age", "Fat", "IAT", "Sprint_30", "SIMS_score", "SIMS_pain", "Srpe_7d_robust", "Matchday", "Srpe_team_avg", "KEB_AB_robust")
  
  data_training <- subset(data_training, select = subset)
  # Generate cross-validation folds index and remove ID
  result$CV_folds<-groupKFold(data_training$ID, k = length(unique(data_training$ID))) 
  result$player_in_training<-length(unique(data_training$ID))
  result$data_training_wids<-data_training
  result$data_training<-select(data_training, !ID)
  result$full_dataset<-data
  result$initial_training_IDs<-training_IDs
  return(result)
}

# Upsampling
rose_upsampling <- list(name = "ROSE",
                        func = function (x, y) {
                          library(ROSE) 
                          dat <- if (is.data.frame(x)) x else as.data.frame(x)
                          dat$.y <- y
                          dat <- ROSE(.y ~ ., data = dat,seed = 42)$data
                          list(x = dat[, !grepl(".y", colnames(dat), fixed = TRUE)], 
                               y = dat$.y)
                        },
                        first = FALSE)

# Tune grid
gbmGrid <-    expand.grid(interaction.depth = 2^(0:4), 
                          n.trees = (1:15)*25, 
                          shrinkage = 0.1, 
                          n.minobsinnode = 10) 

# Feature importance
plot_varimp<-function(model){
  plot_data<-varImp(model)$importance%>%rownames_to_column("Feature")%>%dplyr::rename(Importance=Overall)
  plot_data$Feature<-factor(plot_data$Feature,levels = plot_data$Feature[order(plot_data$Importance)] ,ordered = T)
  result<-ggplot(plot_data,aes(x=Importance,y=Feature)) +geom_segment( aes(x=0, xend=Importance, y=Feature, yend=Feature), color="grey") +
    geom_point( color="blue") +ggplot_theme+scale_x_continuous(expand = c(0,0))
  return (result)
}

# Violin plots (visualizing discrimination)
violin_plot<-function(input_data){
  if ("Crit"%in%names(input_data)){
    result=ggplot(input_data, aes(x=Crit, y=prob_yes, color = Crit))
  }else{
    #freeze colors
    input_data$obs <- factor(input_data$obs, levels = c("Yes", "No"))
    result=ggplot(input_data, aes(x=obs, y=prob_yes, color = obs))
  }
  
  #plot
  result + 
    geom_violin(fill=NA, draw_quantiles = c(0.25, 0.5, 0.75))  + 
    ggtitle("Cross-validation results") + 
    xlab("Day of criterion injury") + 
    ylab("Predicted injury probability") +
    ggplot_theme+ theme(legend.position ="none") +
    scale_y_continuous(limits = c(0,1),expand = c(0,0))
}

# Calibration plots
generate_calibration_plot<-function(input_data,recalibration_data){
  ## Brier score
  if ("CritL"%in%names(input_data)){
    lbl <-"CritL" 
  }else{
    lbl <-"obs_i"
  }
  # Calibration plot for raw probability predictions
  res1=calibration_plot(data = input_data, obs=lbl, pred="prob_yes", title="Calibration plot - Raw probability predictions for training set", xlab = "Probability predictions", ylab = "Observed relative frequency")
  
  # mean raw probability prediction
  prob_raw <- mean(input_data$prob_yes)
  
  # Ancillary: Recalibrating based on proportion of minority class in the training set 
  ## Calculating baseline proportion in the training set
  baseline_tab <- table(recalibration_data$Crit)
  baseline_prob_train <- baseline_tab[2] /(baseline_tab[1]+baseline_tab[2])
  
  ## Recalibrating
  input_data$prob_yes_c <- input_data$prob_yes * (baseline_prob_train / prob_raw)
  
  ## Check
  res2=round(mean(input_data$prob_yes_c - baseline_prob_train))
  
  # Calibration plot after recalibration 
  res3=calibration_plot(data = input_data, obs=lbl, pred="prob_yes_c", title="Calibration plot - Training set predictions after recalibration", xlab = "Probability predictions - recalibrated using relative frequency of minority class in training set", ylab = "Observed relative frequency")
  return(list("calibration"=res1,"recalibration_check"=res2,"recalibrated_calibration"=res3))
}

# Mean for Brier scores
Boot_mean <- function(data, indices){
  d <- data[indices]
  return(mean(d))
}

# Brier scores and confidence interval
compute_brier<-function(input_data){
  result=list()
  ## Brier score
  if ("CritL"%in%names(input_data)){
    input_data$Brier <- (input_data$prob_yes - input_data$CritL)^2
  }else{
    input_data$Brier <- (input_data$prob_yes - input_data$obs_i)^2
  }
  result$Brier_score=mean(input_data$Brier)
  ### Check distribution of Brier scores 
  result$Brier_score_dist=ggplot(input_data, aes(x=Brier)) + geom_density()
  
  ### Bootstrap confidence interval for Brier score 
  set.seed(42)
  results_train <- boot(data = input_data$Brier, statistic = Boot_mean, R=1000)
  result$Brier_bootstrap_trainset=plot(results_train)
  
  result$Brier_ci=boot.ci(boot.out = results_train, conf = 0.95, type=c("norm", "basic", "perc"))
  return(result)
}



# ROC analysis
generate_rocs<-function(input_data){
  result=list()
  # ROC curve with confidence interval
  if ("Crit"%in%names(input_data)){
    lbl <-"Crit" 
  }else{
    lbl <-"obs"
  }
  roc_o <- roc(input_data[[lbl]], input_data$prob_yes)
  ciobj <- ci.se(roc_o, specificities=seq(0, 1, l=25),parallel=T)
  dat.ci <- data.frame(x = as.numeric(rownames(ciobj)),
                       lower = ciobj[, 1],
                       upper = ciobj[, 3])
  
  result$roc_plot<-ggroc(roc_o) + theme_minimal() + geom_abline(slope=1, intercept = 1, linetype = "dashed", alpha=0.7, color = "grey") + coord_equal() + 
    geom_ribbon(data = dat.ci, aes(x = x, ymin = lower, ymax = upper), fill = "steelblue", alpha= 0.2) + ggtitle("Main model - cross validation")
  # ROC-AUC
  result$auc<- auc(roc_o)
  # ROC-AUC confidence interval
  result$auc_ci <- ci.auc(roc_o, conf.level=0.95, parallel=T)
  
  return(result)
}

