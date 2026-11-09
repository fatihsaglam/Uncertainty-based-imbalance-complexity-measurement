library(R6)
library(reticulate)

classifier_NNET <- R6Class(
  "classifier_NNET",
  public = list(
    model = NULL,
    device = NULL,
    class_names = NULL,
    class_pos = NULL,
    torch = NULL,
    
    initialize = function() {
      # torch ilk nesne oluşturulurken yüklenir (source edilirken değil)
      self$torch <- torch
      dev <- if (self$torch$cuda$is_available()) "cuda" else "cpu"
      self$device <- self$torch$device(dev)
    },
    
    fit = function(x, y, hidden_dim = 32L, n_hidden = 2L,
                   activation = "relu", epochs = 50L,
                   batch_size = 64L, lr = 1e-3, seed = NULL) {
      torch <- self$torch; nn <- torch$nn
      
      if (!is.null(seed)) { set.seed(seed); torch$manual_seed(as.integer(seed)) }
      
      # Veri hazırlığı
      x_mat <- as.matrix(x)
      storage.mode(x_mat) <- "double"
      n <- nrow(x_mat); p <- ncol(x_mat)
      
      self$class_names <- levels(y)
      self$class_pos   <- names(which.min(table(y)))
      y_bin <- as.numeric(y == self$class_pos)
      
      # Aktivasyon (orijinaldeki gibi tek örnek tüm katmanlarda paylaşılır; durumsuz olduğu için sorun yok)
      act_fn <- switch(tolower(activation),
                       relu    = nn$ReLU(),
                       lrelu   = nn$LeakyReLU(0.2),
                       tanh    = nn$Tanh(),
                       sigmoid = nn$Sigmoid(),
                       nn$ReLU())
      
      # Ağ: Linear + aktivasyon katmanları, çıkışta tek logit
      layers <- list(); in_f <- p
      for (i in seq_len(n_hidden)) {
        layers[[length(layers) + 1]] <- nn$Linear(as.integer(in_f), as.integer(hidden_dim))
        layers[[length(layers) + 1]] <- act_fn
        in_f <- hidden_dim
      }
      layers[[length(layers) + 1]] <- nn$Linear(as.integer(in_f), 1L)
      model <- do.call(nn$Sequential, unname(layers))
      model$to(self$device)
      
      opt       <- torch$optim$Adam(model$parameters(), lr = lr)
      criterion <- nn$BCEWithLogitsLoss()
      bs        <- as.integer(batch_size)
      
      # Veriyi bir kez tensöre çevir, batch'i torch içinde seç (R-Python geçişlerini azaltır)
      x_t <- torch$from_numpy(np_array(x_mat, dtype = "float32"))$to(self$device)
      y_t <- torch$from_numpy(np_array(matrix(y_bin, ncol = 1), dtype = "float32"))$to(self$device)
      n_batches <- ceiling(n / bs)
      
      # Eğitim döngüsü
      model$train()
      for (epoch in seq_len(epochs)) {
        perm <- torch$randperm(as.integer(n))
        for (b in seq_len(n_batches)) {
          start <- (b - 1L) * bs
          len   <- min(bs, n - start)
          ids <- perm$narrow(0L, as.integer(start), as.integer(len))
          xb  <- x_t$index_select(0L, ids)
          yb  <- y_t$index_select(0L, ids)
          
          opt$zero_grad()
          loss <- criterion(model(xb), yb)
          loss$backward()
          opt$step()
        }
      }
      
      self$model <- model
      invisible(self)
    },
    
    predict = function(x) {
      torch <- self$torch
      self$model$eval()
      
      x_mat <- as.matrix(x); storage.mode(x_mat) <- "double"
      xt <- torch$from_numpy(np_array(x_mat, dtype = "float32"))$to(self$device)
      probs_pos <- as.numeric(torch$sigmoid(self$model(xt))$detach()$cpu()$numpy())
      
      # sadece ikili sınıflama
      prob <- matrix(NA_real_, nrow = length(probs_pos), ncol = 2)
      prob[, which(self$class_names == self$class_pos)] <- probs_pos
      prob[, which(self$class_names != self$class_pos)] <- 1 - probs_pos
      colnames(prob) <- self$class_names
      
      pred <- factor(self$class_names[max.col(prob)], levels = self$class_names)
      list(pred = pred, prob = prob)
    }
  )
)