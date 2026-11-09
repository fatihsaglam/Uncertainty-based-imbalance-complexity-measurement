library(reticulate)

resamplerCls_deepSMOTE <- function(x, y, k = 5L, lr = 0.01, batch_size = 32L,
                                   epochs = 100L, hidden_dim = 16L,
                                   latent_dim = 4L, ovRate = 1, seed = NULL, ...) {
  
  torch <- import("torch")
  nn    <- torch$nn
  nnf   <- nn$functional
  
  if (!is.null(seed)) { set.seed(seed); torch$manual_seed(as.integer(seed)) }
  
  device <- torch$device(if (torch$cuda$is_available()) "cuda" else "cpu")
  to_t   <- function(m) torch$from_numpy(np_array(m, dtype = "float32"))$to(device)
  
  ## ---- Veri hazırlığı ----
  x <- as.matrix(x); storage.mode(x) <- "double"
  p   <- ncol(x)
  h   <- as.integer(hidden_dim)
  lat <- as.integer(latent_dim)
  bs  <- as.integer(batch_size)
  class_names <- levels(y)
  n_classes   <- table(y)
  n_needed    <- round((max(n_classes) - n_classes) * ovRate)
  
  x_syn_list <- list(); y_syn_list <- list()
  
  for (cls in class_names) {
    n_gen <- as.integer(n_needed[cls])
    x_cls <- x[y == cls, , drop = FALSE]
    n     <- nrow(x_cls)
    if (n_gen <= 0L || n < 2L) next
    
    x_t <- to_t(x_cls)
    
    ## ---- Sabit batch'ler (orijinaldeki gibi: bir kez karıştırılır, epoch'lar boyunca aynı kalır) ----
    idx_perm <- torch$randperm(as.integer(n), device = device)
    batches <- lapply(seq(0L, n - 1L, by = bs), function(s) {
      x_t$index_select(0L, idx_perm$narrow(0L, as.integer(s), as.integer(min(bs, n - s))))
    })
    
    ## ---- Encoder / decoder ----
    enc <- nn$Sequential(nn$Linear(as.integer(p), h), nn$ReLU(), nn$Linear(h, lat))$to(device)
    dec <- nn$Sequential(nn$Linear(lat, h), nn$ReLU(), nn$Linear(h, as.integer(p)))$to(device)
    opt <- torch$optim$Adam(nn$ModuleList(list(enc, dec))$parameters(), lr = lr)
    
    ## ---- Eğitim: yeniden oluşturma kaybı (RL) + DeepSMOTE cezası (PL) ----
    enc$train(); dec$train()
    for (epoch in seq_len(epochs)) {
      for (batch in batches) {
        opt$zero_grad()
        
        # Standart yeniden oluşturma
        EB <- enc(batch)
        DB <- dec(EB)
        RL <- nnf$mse_loss(DB, batch)
        
        # Ceza: gözlemleri karıştır, gizli özellikleri permüte et, tekrar çöz
        nb <- as.integer(batch$size(0L))
        CD <- batch$index_select(0L, torch$randperm(nb, device = device))
        ES <- enc(CD)
        PE <- ES$index_select(1L, torch$randperm(lat, device = device))
        DP <- dec(PE)
        PL <- nnf$mse_loss(DP, CD)
        
        TL <- RL$add(PL)
        TL$backward()
        opt$step()
      }
    }
    
    ## ---- Üretim: gizli uzayda SMOTE, sonra decoder ----
    enc$eval(); dec$eval()
    x_latent <- NULL
    with(torch$no_grad(), {
      x_latent <- as.matrix(enc(x_t)$cpu()$numpy())
    })
    
    k_eff  <- min(as.integer(k), n - 1L)
    nn_idx <- FNN::get.knnx(x_latent, x_latent, k = k_eff + 1L)$nn.index[, -1, drop = FALSE]
    
    base <- sample.int(n, n_gen, replace = TRUE)
    nbr  <- nn_idx[cbind(base, sample.int(ncol(nn_idx), n_gen, replace = TRUE))]
    r    <- runif(n_gen)
    x_syn_latent <- x_latent[base, , drop = FALSE] +
      r * (x_latent[nbr, , drop = FALSE] - x_latent[base, , drop = FALSE])
    
    x_syn_cls <- NULL
    with(torch$no_grad(), {
      x_syn_cls <- as.matrix(dec(to_t(x_syn_latent))$cpu()$numpy())
    })
    colnames(x_syn_cls) <- colnames(x)
    
    x_syn_list[[cls]] <- x_syn_cls
    y_syn_list[[cls]] <- factor(rep(cls, n_gen), levels = levels(y))
  }
  
  if (length(x_syn_list) == 0L) {
    x_syn <- matrix(numeric(0), nrow = 0, ncol = p, dimnames = list(NULL, colnames(x)))
    y_syn <- factor(character(0), levels = levels(y))
  } else {
    x_syn <- do.call(rbind, unname(x_syn_list))
    y_syn <- do.call(c, unname(y_syn_list))
  }
  
  list(x_new = rbind(x, x_syn), y_new = c(y, y_syn),
       x_syn = x_syn, y_syn = y_syn)
}