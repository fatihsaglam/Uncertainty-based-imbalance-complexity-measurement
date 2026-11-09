#' Model Factory for Machine Learning Benchmarks
#'
#' This function acts as a centralized dispatcher. It takes a simple string 
#' representing a model name and returns the corresponding initialized R6 
#' classifier object. 
#'
#' @param model_name A character string specifying the classifier to instantiate.
#' @return An initialized R6 object with standard $fit() and $predict() methods.
#' @export
get_classifier <- function(model_name) {
  # 1. Standardize input to prevent case-sensitivity or spacing errors
  clean_name <- toupper(trimws(model_name))
  
  # 2. Dispatch to the correct R6 generator
  model <- switch(clean_name,
                  "NNET"       = classifier_NNET$new(),
                  "BNNET"      = classifier_BNNET$new(),
                  "KANNET"     = classifier_KANNET$new(),
                  "GBM"        = classifier_GBM$new(),
                  "RF"         = classifier_RF$new(),
                  "CART"       = classifier_CART$new(),
                  "BART"       = classifier_BART$new(),
                  "ADABOOST"   = classifier_AdaBoost$new(),
                  "SVMLINEAR"  = classifier_SVMLinear$new(),
                  "SVMRADIAL"  = classifier_SVMRadial$new(),
                  "MARS"       = classifier_MARS$new(),
                  "KNN"        = classifier_KNN$new(),
                  "LDA"        = classifier_LDA$new(),
                  "QDA"        = classifier_QDA$new(),
                  "NAIVEBAYES" = classifier_NaiveBayes$new(),
                  "LOGREG"     = classifier_LogReg$new(),
                  "ELASTICNET" = classifier_ElasticNet$new(),
                  
                  # 3. Fallback error if the user requests a model that doesn't exist
                  stop(
                    sprintf(
                      "Classifier '%s' not recognized. Please check the spelling or add it to factory.R.", 
                      model_name
                    ), 
                    call. = FALSE
                  )
  )
  
  return(model)
}

#' List Available Classifiers
#' 
#' A helper function to quickly see all valid strings accepted by get_classifier()
#' @export
list_classifiers <- function() {
  c("NNET", "BNNET", "KANNET", "GBM", "RF", "CART", "BART", "ADABOOST", 
    "SVMLINEAR", "SVMRADIAL", "MARS", "KNN", "LDA", "QDA", "NAIVEBAYES", 
    "LOGREG", "ELASTICNET")
}

#' Resampler Factory for Machine Learning Benchmarks
#'
#' Takes a string representing a resampler name and returns a list 
#' containing the resampling function and its hyperparameter grid.
#'
#' @param resampler_name A character string specifying the resampler.
#' @return A list with `$fun` (the function) and `$hypGrid` (the grid).
#' @export
get_resampler <- function(resampler_name) {
  # Standardize input
  clean_name <- trimws(resampler_name)
  
  resampler <- switch(clean_name,
                      "NORES" = list(
                        fun = function(x, y, ...) { list(x_new = x, y_new = y) },
                        hypGrid = expand.grid(dummy = 1)[, 0, drop = FALSE]
                      ),
                      "ROS" = list(
                        fun = SMOTEWB::ROS,
                        hypGrid = expand.grid(ovRate = c(0.1, 0.25, 0.5, 1, 1.5, 2))
                      ),
                      "SMOTE" = list(
                        fun = SMOTEWB::SMOTE,
                        hypGrid = expand.grid(k = c(3, 5, 7, 9), ovRate = c(0.1, 0.25, 0.5, 1, 1.5, 2))
                      ),
                      "RWO" = list(
                        fun = SMOTEWB::RWO,
                        hypGrid = expand.grid(ovRate = c(0.1, 0.25, 0.5, 1, 1.5, 2))
                      ),
                      "ROSE" = list(
                        fun = SMOTEWB::ROSE,
                        hypGrid = expand.grid(h = c(0.5, 1, 1.5), ovRate = c(0.1, 0.25, 0.5, 1, 1.5, 2))
                      ),
                      "GSMOTE" = list(
                        fun = SMOTEWB::GSMOTE,
                        hypGrid = expand.grid(
                          k = c(3, 5, 7, 9),
                          alpha_trunc = c(-1, -0.5, 0, 0.5, 1),
                          alpha_def = c(0, 0.5, 1),
                          ovRate = c(0.1, 0.25, 0.5, 1, 1.5, 2)
                        )
                      ),
                      "SMOTEWB" = list(
                        fun = SMOTEWB::SMOTEWB,
                        hypGrid = expand.grid(n_weak_classifier = c(50, 100), ovRate = c(0.1, 0.25, 0.5, 1, 1.5, 2))
                      ),
                      "SyMProD" = list(
                        fun = resamplerCls_SyMProD, # Ensure this is loaded from R/resamplers/
                        hypGrid = expand.grid(
                          NT = c(2.5, 3),
                          CT = c(0, 0.5, 1),
                          K = c(3, 5, 7),
                          M = c(3, 5, 7),
                          ovRate = c(0.1, 0.25, 0.5, 1, 1.5, 2)
                        )
                      ),
                      "GQEO" = list(
                        fun = resamplerCls_GQEO, # Ensure this is loaded from R/resamplers/
                        hypGrid = expand.grid(k = c(3, 5, 7, 9), ovRate = c(0.1, 0.25, 0.5, 1, 1.5, 2))
                      ),
                      "EBSMOTE" = list(
                        fun = resamplerCls_EBSMOTE, # Ensure this is loaded from R/resamplers/
                        hypGrid = expand.grid(
                          m_alpha = c(3, 5, 7, 9),
                          m_beta = c(3, 5, 7, 9),
                          lambda = c(2.5, 3, 3.5, 4),
                          v_alpha = c(3, 5, 7, 9),
                          w_alpha = c(3, 5, 7, 9),
                          expand_alpha = c(-1, -0.5, 0, 0.5, 1),
                          ovRate = c(0.1, 0.25, 0.5, 1, 1.5, 2)
                        )
                      ),
                      "deepSMOTE" = list(
                        fun = resamplerCls_deepSMOTE, # Ensure this is loaded from R/resamplers/
                        hypGrid = expand.grid(
                          k = c(3, 5, 7, 9),
                          epochs = c(50, 100, 200),
                          batch_size = c(128L),
                          hidden_dim = c(8L, 16L, 32L, 64L),
                          latent_dim = c(8L, 16L, 32L, 64L),
                          ovRate = c(0.1, 0.25, 0.5, 1, 1.5, 2)
                        )
                      ),
                      "LRDO" = list(
                        fun = resamplerCls_LRDO, # Ensure this is loaded from R/resamplers/
                        hypGrid = expand.grid(k = c(3, 5, 7, 9), ovRate = c(0.1, 0.25, 0.5, 1, 1.5, 2))
                      ),
                      "GAN" = list(
                        fun = resamplerCls_GAN, # Ensure this is loaded from R/resamplers/
                        hypGrid = expand.grid(
                          lr = c(0.001, 0.01, 0.02),
                          epochs = c(200L, 400L, 800L),
                          batch_size = c(128L),
                          hidden_dim = c(16L, 32L),
                          noise_dim = c(16L, 32L),
                          ovRate = c(0.1, 0.25, 0.5, 1, 1.5, 2)
                        )
                      ),
                      "AWGAN" = list(
                        fun = resamplerCls_AWGAN, 
                        hypGrid = expand.grid(
                          lr = c(0.001, 0.01, 0.02),
                          epochs = c(200L, 400L, 800L),
                          batch_size = c(128L),
                          hidden_dim = c(16L, 32L),
                          noise_dim = c(16L, 32L),
                          lambda_gp = c(0.05, 0.1, 0.2),
                          similarity_threshold = c(0.75, 0.80, 0.85),
                          ovRate = c(0.1, 0.25, 0.5, 1, 1.5, 2)
                        )
                      ),
                      "BiSMOTE" = list(
                        fun = resamplerCls_BiSMOTE, # Ensure this is loaded from R/resamplers/
                        hypGrid = expand.grid(
                          k = c(3L, 5L, 7L, 9L),
                          ovRate = c(0.1, 0.25, 0.5, 1, 1.5, 2)
                        )
                      ),
                      "LEAO" = list(
                        fun = resamplerCls_LEAO, # Ensure this is loaded from R/resamplers/
                        hypGrid = expand.grid(
                          iters = c(25L, 50L, 100L),
                          alpha = c(0.5, 1.0, 2.0),
                          prob_type = c("bagging", "logistic", "knn", "nb"),
                          k = c(3L, 5L, 7L),
                          stop_prob = c(0.1, 0.3, 0.5),
                          step_max = c(0.05, 0.10, 0.20), stringsAsFactors = FALSE
                        )
                      ),
                      "MCSMOTE" = list(
                        fun = resamplerCls_MCSMOTE, # Ensure this is loaded from R/resamplers/
                        hypGrid = expand.grid(
                          thresh_qntl = c(0.001, 0.01, 0.05, 0.1),
                          ovRate = c(0.1, 0.25, 0.5, 1, 1.5, 2)
                        )
                      ),
                      "DAGAN" = list(
                        fun = resamplerCls_DAGAN, 
                        hypGrid = expand.grid(
                          lr = c(0.005, 0.01, 0.02),
                          epochs = c(200L, 400L, 800L),
                          batch_size = c(128L),
                          hidden_dim = c(16L, 32L),
                          noise_dim = c(16L, 32L),
                          lambda_mmd = c(0.05, 0.1, 0.2),
                          ovRate = c(0.1, 0.25, 0.5, 1, 1.5, 2)
                        )
                      ),
                      "SMOTifiedGAN" = list(
                        fun = resamplerCls_SMOTifiedGAN, 
                        hypGrid = expand.grid(
                          lr = c(0.001, 0.01, 0.02),
                          epochs = c(500L, 1000L, 2000L),
                          batch_size = c(128L),
                          hidden_dim = c(64L, 128L, 256L),
                          noise_dim = c(4L, 8L, 16L),
                          k = c(3L, 5L, 7L),
                          ovRate = c(0.1, 0.25, 0.5, 1, 1.5, 2)
                        )
                      ),
                      "WGANGP" = list(
                        fun = resamplerCls_WGANGP, 
                        hypGrid = expand.grid(
                          lr = c(1e-4, 1e-3, 5e-3),        # Centered on 1e-3
                          epochs = c(100L, 200L, 500L),    # Centered on 100L
                          batch_size = c(128L),  # Centered on 64L
                          hidden_dim = c(16L, 32L, 64L),   # Centered on 32L
                          noise_dim = c(8L, 16L, 32L),     # Centered on 16L
                          lambda_gp = c(0.1, 1, 10),       # Centered on 1
                          critic_iters = c(5L),
                          ovRate = c(0.1, 0.25, 0.5, 1, 1.5, 2)
                        )
                      ),
                      "VAE" = list(
                        fun = resamplerCls_VAE, 
                        hypGrid = expand.grid(
                          lr = c(1e-5, 1e-4, 1e-3),        # Centered on your 0.0001
                          epochs = c(200L, 500L, 1000L),   # Centered on 500L
                          batch_size = c(128L),  # Centered on 64L
                          hidden_dim = c(16L, 32L, 64L),   # Centered on 32L
                          latent_dim = c(2L, 4L, 8L),      # Centered on 4L
                          ovRate = c(0.1, 0.25, 0.5, 1, 1.5, 2)
                        )
                      ),
                      "ADA_INCVAE" = list(
                        fun = resamplerCls_ADA_INCVAE, 
                        hypGrid = expand.grid(
                          lr = c(1e-5, 1e-4, 1e-3),        # Centered on 1e-4
                          epochs = c(100L, 200L, 500L),    # Centered on 200L
                          batch_size = c(128L),       # Centered on 128L
                          hidden_dim = c(32L, 64L, 128L),  # Centered on 64L
                          latent_dim = c(2L, 4L, 8L),      # Centered on 4L
                          k = c(3L, 5L, 7L),               # Centered on 5L
                          ovRate = c(0.1, 0.25, 0.5, 1, 1.5, 2)
                        )
                      ),
                      "SSNNetO" = list(
                        fun = resamplerCls_SSNNetO, 
                        hypGrid = expand.grid(
                          lr          = c(1e-4, 1e-3, 5e-3),        # Centered on 0.001
                          epochs      = c(25L, 50L, 100L),          # 50L is your sweet spot
                          batch_size  = c(128L),         # 128L for speed/stability
                          hidden_dim  = c(32L, 64L),           # Centered on 32L
                          kl_weight   = c(1e-8, 1e-7, 1e-6, 1e-5),        # Very low to prioritize Hinge
                          init_logvar = c(-2.0),        # Tightness of initial shell
                          ovRate      = c(0.1, 0.25, 0.5, 1, 1.5, 2)
                        )
                      ),
                      
                      stop(sprintf("Resampler '%s' not recognized. Please check factory.R", resampler_name), call. = FALSE)
  )
  
  return(resampler)
}

#' List Available Resamplers
#' @export
list_resamplers <- function() {
  c("NORES", "ROS", "SMOTE", "RWO", "ROSE", "GSMOTE", "SMOTEWB", 
    "SyMProD", "GQEO", "EBSMOTE", "deepSMOTE", "LRDO", "GAN", "AWGAN",
    "BiSMOTE", "LEAO", "MCSMOTE", "DAGAN", "SMOTifiedGAN", "WGANGP", "VAE",
    "ADA_INCVAE", "SSNNetO")
}

#' Metric Factory
#' @param metric_names A character vector of metric names (e.g., c("ROCAUC", "F1"))
#' @return A subsetted list of metric functions
get_metrics <- function(metric_names) {
  # Validation
  invalid <- setdiff(metric_names, names(metricCls_all))
  if (length(invalid) > 0) {
    stop(paste("Metrics not found:", paste(invalid, collapse = ", ")))
  }
  
  # Return only the requested functions
  return(metricCls_all[metric_names])
}

## =====================================================
## COMPLEXITY MEASURES (literatür + önerilen)
## Yükleme anında hiçbir ölçü nesnesine bakılmaz; her şey çağrı anında çözülür.
## Gereken dosyalar: complexity_literature.R (içinden complexity_proposed.R gelir)
## =====================================================
DIST_MEASURES <- c("N1", "N2", "N3", "N3_bal", "LSC", "UNSAFE_MIN", "RARE_OUT")   # mesafe matrisi gerektirenler

# Dahili: ölçü kaydını çağrı anında kurar
.cm_registry <- function() {
  needed <- c("LITERATURE_MEASURES", "LITERATURE_INFO",
              "PROPOSED_MEASURES",   "PROPOSED_INFO",
              "complexityCls_context", "complexityCls_uncertainty")
  found <- vapply(needed, function(nm) !is.null(get0(nm)), logical(1))
  if (!all(found)) {
    stop(sprintf("Complexity objects not found: %s. Source the complexity files first.",
                 paste(needed[!found], collapse = ", ")), call. = FALSE)
  }
  list(measures = c(LITERATURE_MEASURES, PROPOSED_MEASURES),
       info     = rbind(cbind(LITERATURE_INFO, source = "literatür"),
                        cbind(PROPOSED_INFO,   source = "önerilen")))
}

# Dahili: kullanıcı girdisini (büyük/küçük harf ve boşluk fark etmeksizin) kanonik ölçü adlarına çevirir
.resolve_complexity_names <- function(names_in, valid) {
  clean <- trimws(as.character(names_in))
  idx <- match(toupper(clean), toupper(valid))
  if (anyNA(idx)) {
    stop(sprintf("Complexity measure(s) not recognized: %s. See list_complexities() for valid names.",
                 paste(clean[is.na(idx)], collapse = ", ")), call. = FALSE)
  }
  unique(valid[idx])
}

#' Complexity Measure Factory
#'
#' Takes measure names and returns the corresponding measure functions.
#'
#' @param complexity_names A character vector of measure names (e.g., c("IR", "N3_bal", "EU_min")).
#' @return A named list of functions, each with signature f(X, y, ctx = NULL).
#' @export
get_complexities <- function(complexity_names) {
  reg <- .cm_registry()
  reg$measures[.resolve_complexity_names(complexity_names, names(reg$measures))]
}

#' List Available Complexity Measures
#' @export
list_complexities <- function() {
  names(.cm_registry()$measures)
}

#' Complexity Measure Details
#'
#' Table with group, reference, source (literature or proposed) and
#' computational requirements of every measure.
#' @return A data.frame.
#' @export
info_complexities <- function() {
  reg <- .cm_registry()
  info <- reg$info
  info$needs_dist <- info$name %in% DIST_MEASURES
  info$needs_unc  <- info$name %in% names(PROPOSED_MEASURES)
  rownames(info) <- NULL
  info
}

#' Compute Complexity Measures
#'
#' Computes the requested measures in one call. The distance matrix and the RF
#' uncertainty estimates are computed once and shared across measures.
#'
#' @param X A numeric data.frame or matrix of features.
#' @param y A binary label vector (the minority class is recoded to 1 internally).
#' @param which Measure names (see list_complexities()). Default: all measures.
#' @param ctx An optional precomputed context from cm_context().
#' @param scale_type "zscore" or "minmax" (used for distance-based measures).
#' @param proposed_args A list of arguments passed to rf_entropy_uncertainty().
#' @param seed Seed for randomized measures (N4).
#' @param verbose If TRUE, prints the reason when a measure fails (returned as NA).
#' @return A named numeric vector.
#' @export
compute_complexities <- function(X, y, which = list_complexities(), ctx = NULL,
                                 scale_type = "zscore", proposed_args = list(),
                                 seed = NULL, verbose = FALSE) {
  reg       <- .cm_registry()
  which     <- .resolve_complexity_names(which, names(reg$measures))
  need_unc  <- any(which %in% names(PROPOSED_MEASURES))
  need_dist <- any(which %in% DIST_MEASURES)
  
  if (is.null(ctx)) {
    ctx <- complexityCls_context(X, y, scale_type = scale_type, with_distances = need_dist,
                      with_proposed = need_unc, proposed_args = proposed_args)
  } else {
    if (need_dist && is.null(ctx$D)) stop("ctx mesafe içermiyor (with_distances = TRUE gerekir)", call. = FALSE)
    if (need_unc && is.null(ctx$unc))
      ctx$unc <- do.call(complexityCls_uncertainty, c(list(X = ctx$X, y = ctx$y), proposed_args))
  }
  
  out <- setNames(rep(NA_real_, length(which)), which)
  for (nm in which) {
    args <- list(X = ctx$X, y = ctx$y, ctx = ctx)
    if (nm == "N4" && !is.null(seed)) args$seed <- seed
    out[nm] <- tryCatch(as.numeric(do.call(reg$measures[[nm]], args)),
                        error = function(e) {
                          if (verbose) message(nm, ": ", conditionMessage(e))
                          NA_real_
                        })
  }
  out
}

#' Data Generator Factory
#'
#' Takes a string with a data generator (family) name and returns a list
#' containing the generating function and its family specification.
#'
#' @param generator_name A single character string (e.g., "ovl").
#' @return A list with `$fun` (f(n0, n1, P, noise_eta, noise_sd) -> data.frame with a `y` column)
#'   and `$spec` (the family definition).
#' @export
get_data_generator <- function(generator_name) {
  clean_name <- trimws(as.character(generator_name))
  if (length(clean_name) != 1L) {
    stop("get_data_generator() takes a single name; use lapply(names, get_data_generator) for several.",
         call. = FALSE)
  }
  valid <- names(dataGeneratorCls_all)
  idx <- match(tolower(clean_name), tolower(valid))
  if (is.na(idx)) {
    stop(sprintf("Data generator '%s' not recognized. Valid: %s",
                 clean_name, paste(valid, collapse = ", ")), call. = FALSE)
  }
  list(fun = dataGeneratorCls_all[[idx]], spec = DATAGEN_SPECS[[valid[idx]]])
}

#' List Available Data Generators
#' @export
list_data_generators <- function() {
  names(dataGeneratorCls_all)
}