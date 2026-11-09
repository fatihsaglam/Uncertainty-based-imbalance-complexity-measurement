library(R6)
library(reticulate)

classifier_BNNET <- R6Class(
  "classifier_BNNET",
  public = list(
    model = NULL,
    device = NULL,
    class_names = NULL,
    class_pos = NULL,
    torch = NULL,
    bnn = NULL,
    
    initialize = function() {
      # torch ve torchbnn ilk nesne oluşturulurken yüklenir (source edilirken değil)
      self$torch  <- torch
      self$bnn    <- torchbnn
      self$device <- self$torch$device("cpu")
    },
    
    fit = function(x, y, hidden_dim = 32L, n_hidden = 2L,
                   prior_mu = 0.0, prior_sigma = 0.1,
                   kl_weight = 0.1, epochs = 50L,
                   batch_size = 64L, lr = 1e-2, seed = NULL) {
      torch <- self$torch; bnn <- self$bnn; nn <- torch$nn
      
      if (!is.null(seed)) { set.seed(seed); torch$manual_seed(as.integer(seed)) }
      
      x_mat <- as.matrix(x)
      storage.mode(x_mat) <- "double"
      n <- nrow(x_mat); p <- ncol(x_mat)
      
      self$class_names <- levels(y)
      self$class_pos   <- names(which.min(table(y)))
      y_bin <- as.numeric(y == self$class_pos)
      
      ## ---- Ağ: BayesLinear + ReLU katmanları, çıkışta tek logit ----
      layers <- list(); in_f <- p
      for (i in seq_len(n_hidden)) {
        layers[[length(layers) + 1]] <- bnn$BayesLinear(
          prior_mu = as.numeric(prior_mu), prior_sigma = as.numeric(prior_sigma),
          in_features = as.integer(in_f), out_features = as.integer(hidden_dim))
        layers[[length(layers) + 1]] <- nn$ReLU()
        in_f <- hidden_dim
      }
      layers[[length(layers) + 1]] <- bnn$BayesLinear(
        prior_mu = as.numeric(prior_mu), prior_sigma = as.numeric(prior_sigma),
        in_features = as.integer(in_f), out_features = 1L)
      model <- do.call(nn$Sequential, unname(layers))
      model$to(self$device)
      
      opt     <- torch$optim$Adam(model$parameters(), lr = lr)
      ce_loss <- nn$BCEWithLogitsLoss()
      kl_loss <- bnn$BKLLoss(reduction = "mean", last_layer_only = FALSE)
      klw     <- as.numeric(kl_weight)
      bs      <- as.integer(batch_size)
      
      ## ---- Veriyi bir kez tensöre çevir ----
      x_t <- torch$from_numpy(np_array(x_mat, dtype = "float32"))$to(self$device)
      y_t <- torch$from_numpy(np_array(matrix(y_bin, ncol = 1), dtype = "float32"))$to(self$device)
      n_batches <- ceiling(n / bs)
      
      ## ---- Eğitim döngüsü ----
      model$train()
      for (epoch in seq_len(epochs)) {
        perm <- torch$randperm(as.integer(n))            # karıştırma torch içinde
        for (b in seq_len(n_batches)) {
          start <- (b - 1L) * bs
          len   <- min(bs, n - start)
          ids <- perm$narrow(0L, as.integer(start), as.integer(len))
          xb  <- x_t$index_select(0L, ids)
          yb  <- y_t$index_select(0L, ids)
          
          opt$zero_grad()
          l_ce <- ce_loss(model(xb), yb)
          l_kl <- kl_loss(model)
          loss <- l_ce$add(l_kl$mul(klw))                # veri olabilirliği + KL maliyeti
          loss$backward()
          opt$step()
        }
      }
      
      self$model <- model
      invisible(self)
    },
    
    predict = function(x, n_draws = 50L) {
      torch <- self$torch
      self$model$train()   # Bayesci katmanlar ağırlık örneklemek için train modunda kalmalı
      
      x_mat <- as.matrix(x); storage.mode(x_mat) <- "double"
      xt <- torch$from_numpy(np_array(x_mat, dtype = "float32"))$to(self$device)
      
      draws <- vapply(seq_len(n_draws), function(i) {
        as.numeric(torch$sigmoid(self$model(xt))$detach()$cpu()$numpy())
      }, numeric(nrow(x_mat)))
      avg_probs_pos <- rowMeans(draws)
      
      # sadece ikili sınıflama
      prob <- matrix(NA_real_, nrow = length(avg_probs_pos), ncol = 2)
      prob[, which(self$class_names == self$class_pos)] <- avg_probs_pos
      prob[, which(self$class_names != self$class_pos)] <- 1 - avg_probs_pos
      colnames(prob) <- self$class_names
      pred <- factor(self$class_names[max.col(prob)], levels = self$class_names)
      
      list(pred = pred, prob = prob)
    }
  )
)

