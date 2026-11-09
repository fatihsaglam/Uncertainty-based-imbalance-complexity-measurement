# ==============================================================================
# global.R
# ==============================================================================
# 1. Load Libraries
library(reticulate)
torch <- import("torch")
torchbnn <- import("torchbnn")
kan <- import("kan", convert = TRUE)
nn <- import("torch.nn", convert = TRUE)
library(shiny)
library(shinydashboard)
library(DT)
library(R6)          # Required for your Preprocessing and Classifier classes
library(ggplot2)
library(SMOTEWB)     # Required for ROS, SMOTE, RWO, ROSE, GSMOTE, SMOTEWB
library(randomForest)# Required if you use RF
library(caret)       # Often required for data splitting or dummy variables
library(yardstick)
library(kernlab)
library(crayon)
library(openxlsx)

source("R/utils/metrics.R")
source("R/factory.R")
# source("R/datasets.R")
print.datasets_binary <- function(x, ...) {
  print(attr(x, "info"))
}

datasets_binary <- readRDS(file = "data/datasets_binary.rds")
dataset_names <- names(datasets_binary)
datasets_info <- attr(datasets_binary, "info")

# 3. Source all R files recursively
# We find all .R files in the R directory and its subfolders
r_files <- list.files("R", pattern = "\\.[Rr]$", recursive = TRUE, full.names = TRUE)

# Filter out the script itself if it's named something similar, 
# though usually app.R is in the root and won't be caught here.
invisible(lapply(r_files, \(r_file) {source(r_file)}))

## =====================================================
## zz_registry.R -- bileşenlerin fonksiyon listeleri
## Bu dosya EN SON yüklenmeli (ad "zz" ile başlıyor). Metrikler için metricCls_all zaten
## R/utils/metrics.R içinde tanımlı, burada tekrar yok.
## =====================================================

## ---------- Resamplers ----------
resamplerCls_NORES <- function(x, y, ...) list(x_new = x, y_new = y)    # factory'deki satır içi tanımla aynı

resamplerCls_all <- list(
  NORES        = resamplerCls_NORES,
  ROS          = SMOTEWB::ROS,
  SMOTE        = SMOTEWB::SMOTE,
  RWO          = SMOTEWB::RWO,
  ROSE         = SMOTEWB::ROSE,
  GSMOTE       = SMOTEWB::GSMOTE,
  SMOTEWB      = SMOTEWB::SMOTEWB,
  SyMProD      = resamplerCls_SyMProD,
  GQEO         = resamplerCls_GQEO,
  EBSMOTE      = resamplerCls_EBSMOTE,
  deepSMOTE    = resamplerCls_deepSMOTE,
  LRDO         = resamplerCls_LRDO,
  GAN          = resamplerCls_GAN,
  AWGAN        = resamplerCls_AWGAN,
  BiSMOTE      = resamplerCls_BiSMOTE,
  LEAO         = resamplerCls_LEAO,
  MCSMOTE      = resamplerCls_MCSMOTE,
  DAGAN        = resamplerCls_DAGAN,
  SMOTifiedGAN = resamplerCls_SMOTifiedGAN,
  WGANGP       = resamplerCls_WGANGP,
  VAE          = resamplerCls_VAE,
  ADA_INCVAE   = resamplerCls_ADA_INCVAE,
  SSNNetO      = resamplerCls_SSNNetO
)

## ---------- Classifiers (R6 generator nesneleri; örnek için $new()) ----------
classifier_all <- list(
  NNET       = classifier_NNET,
  BNNET      = classifier_BNNET,
  KANNET     = classifier_KANNET,
  GBM        = classifier_GBM,
  RF         = classifier_RF,
  CART       = classifier_CART,
  BART       = classifier_BART,
  ADABOOST   = classifier_AdaBoost,
  SVMLINEAR  = classifier_SVMLinear,
  SVMRADIAL  = classifier_SVMRadial,
  MARS       = classifier_MARS,
  KNN        = classifier_KNN,
  LDA        = classifier_LDA,
  QDA        = classifier_QDA,
  NAIVEBAYES = classifier_NaiveBayes,
  LOGREG     = classifier_LogReg,
  ELASTICNET = classifier_ElasticNet
)

## ---------- Complexity measures (literatür + önerilen; tek kaynak: o iki liste) ----------
complexityCls_all <- c(LITERATURE_MEASURES, PROPOSED_MEASURES)

## ---------- Data generators (her aile bir fonksiyon) ----------
dataGeneratorCls_all <- setNames(lapply(DATAGEN_FAMILIES, dataGeneratorCls_makeFamily),
                                 DATAGEN_FAMILIES)

# 4. Startup Validation & Console Messaging
message("--- Initializing ---")

# Check Datasets (from R/datasets.R)
if (exists("datasets_binary")) {
  message(sprintf("> Datasets loaded: %d", length(datasets_binary)))
} else {
  warning("> WARNING: datasets_binary not found. Check R/datasets.R")
}

# Check Factory Functions (from R/factory.R)
if (exists("list_resamplers") && exists("list_classifiers")) {
  message(sprintf("> Resamplers available: %d", length(list_resamplers())))
  message(sprintf("> Classifiers available: %d", length(list_classifiers())))
}

# Check Metrics (from R/utils/metrics.R)
if (exists("metricCls_all")) {
  message(sprintf("> Metrics registered: %d", length(names(metricCls_all))))
}

message("--- System Ready ---")