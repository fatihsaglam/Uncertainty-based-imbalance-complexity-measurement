library(reticulate)

resamplerCls_deepMIMSMOTE <- function(x, y, k = 5L, lr = 0.001, batch_size = 32L,
                                      epochs = 100L, hidden_dim = 16L, latent_dim = 4L,
                                      ovRate = 1, seed = NULL, ...) {
  
  nnf   <- nn$functional
  
  if (!is.null(seed)) { set.seed(seed); torch$manual_seed(as.integer(seed)) }
  to_t <- function(m) torch$from_numpy(np_array(m, dtype = "float32"))
  
  ## ---- Veri hazırlığı ----
  x <- as.matrix(x); storage.mode(x) <- "double"
  p <- ncol(x)
  if (is.null(colnames(x))) colnames(x) <- paste0("V", seq_len(p))
  class_names <- levels(y)
  n_classes   <- sapply(class_names, function(m) sum(y == m))
  k_class     <- length(class_names)
  n_needed    <- round((max(n_classes) - n_classes) * ovRate)
  h   <- as.integer(hidden_dim)
  lat <- as.integer(latent_dim)
  bs  <- as.integer(batch_size)
  
  ## ---- Eksik veri: MICE ile doldur + eksiklik göstergeleri (MIM) ----
  x_ind <- is.na(x) * 1
  if (anyNA(x)) {
    dat    <- data.frame(x, class_label = y)
    m_mice <- mice::mice(dat, printFlag = FALSE, seed = if (is.null(seed)) NA else seed)
    x_imp  <- as.matrix(mice::complete(m_mice)[, seq_len(p), drop = FALSE])
  } else {
    x_imp <- x                                   # eksik yoksa MICE veriyi değiştirmez
  }
  x_mim <- cbind(x_imp, x_ind)
  colnames(x_mim) <- c(colnames(x), paste0(colnames(x), "_miss"))
  input_dim <- as.integer(ncol(x_mim))
  
  x_syn_list <- list(); y_syn_list <- list()
  
  for (i_class in seq_len(k_class)) {
    cls   <- class_names[i_class]
    n_gen <- as.integer(n_needed[i_class])
    x_main  <- x_mim[y == cls, , drop = FALSE]
    x_other <- x_mim[y != cls, , drop = FALSE]
    n_main  <- nrow(x_main)
    if (n_gen <= 0L || n_main < 2L) next
    
    x_t <- to_t(x_main)
    
    ## ---- Sabit batch'ler (orijinaldeki gibi: bir kez karıştırılır) ----
    idx_perm <- torch$randperm(as.integer(n_main))
    batches <- lapply(seq(0L, n_main - 1L, by = bs), function(s) {
      x_t$index_select(0L, idx_perm$narrow(0L, as.integer(s), as.integer(min(bs, n_main - s))))
    })
    
    ## ---- Encoder / decoder ----
    enc <- nn$Sequential(nn$Linear(input_dim, h), nn$ReLU(), nn$Linear(h, lat))
    dec <- nn$Sequential(nn$Linear(lat, h), nn$ReLU(), nn$Linear(h, input_dim))
    opt <- torch$optim$RMSprop(nn$ModuleList(list(enc, dec))$parameters(), lr = lr)
    
    ## ---- Eğitim: yeniden oluşturma kaybı (RL) + gizli özellik permütasyon cezası (PL) ----
    for (epoch in seq_len(epochs)) {
      for (batch in batches) {
        opt$zero_grad()
        EB <- enc(batch)
        DB <- dec(EB)
        RL <- nnf$mse_loss(DB, batch)
        
        PE <- EB$index_select(1L, torch$randperm(lat))      # gizli özellikleri permüte et
        DP <- dec(PE)
        PL <- nnf$mse_loss(DP, batch)
        
        TL <- RL$add(PL)
        TL$backward()
        opt$step()
      }
    }
    
    ## ---- Gizli uzayda SMOTE ----
    z_main <- as.matrix(enc(x_t)$detach()$numpy())
    k_eff  <- min(as.integer(k), n_main - 1L)
    nn_idx <- FNN::get.knnx(z_main, z_main, k = k_eff + 1L)$nn.index[, -1, drop = FALSE]
    
    base <- sample.int(n_main, n_gen, replace = TRUE)
    nbr  <- nn_idx[cbind(base, sample.int(ncol(nn_idx), n_gen, replace = TRUE))]
    r    <- runif(n_gen)
    z_syn <- z_main[nbr, , drop = FALSE] - r * (z_main[nbr, , drop = FALSE] - z_main[base, , drop = FALSE])
    
    x_dec <- as.matrix(dec(to_t(z_syn))$detach()$numpy())
    
    ## ---- Ayarlama adımı: en yakın azınlık noktasına doğru, karşı sınıf mesafesi yarıçapı içinde kaydır ----
    d_main2other <- FNN::get.knnx(data = x_other, query = x_main, k = 1)$nn.dist[, 1]
    m_s    <- FNN::get.knnx(data = x_main, query = x_dec, k = 1)
    i_near <- m_s$nn.index[, 1]
    d_s    <- m_s$nn.dist[, 1]
    radii  <- d_main2other[i_near]
    
    lo <- (d_s - radii) / d_s
    hi <- (d_s + radii) / d_s
    rr <- lo + runif(n_gen) * (hi - lo)
    rr[d_s == 0] <- 0                                        # sentetik nokta mevcut bir noktayla çakışıyorsa kaydırma yok
    
    x_final <- x_dec + rr * (x_main[i_near, , drop = FALSE] - x_dec)
    colnames(x_final) <- colnames(x_mim)
    
    x_syn_list[[cls]] <- x_final
    y_syn_list[[cls]] <- factor(rep(cls, n_gen), levels = class_names)
  }
  
  if (length(x_syn_list) == 0L) {
    x_syn <- matrix(numeric(0), nrow = 0, ncol = input_dim, dimnames = list(NULL, colnames(x_mim)))
    y_syn <- factor(character(0), levels = class_names)
  } else {
    x_syn <- do.call(rbind, unname(x_syn_list))
    y_syn <- do.call(c, unname(y_syn_list))
  }
  
  x_new <- rbind(x_imp, x_syn[, seq_len(p), drop = FALSE])
  colnames(x_new) <- colnames(x)
  
  list(x_new = x_new, y_new = c(y, y_syn), x_syn = x_syn, y_syn = y_syn)
}