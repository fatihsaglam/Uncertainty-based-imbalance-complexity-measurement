library(reticulate)

resamplerCls_SMOTifiedGAN <- function(x, y, lr = 0.00001, batch_size = 128L, epochs = 1000L,
                                      hidden_dim = 128L, noise_dim = 16L, k = 5L,
                                      ovRate = 1, seed = NULL, ...) {

  if (!is.null(seed)) { set.seed(seed); torch$manual_seed(as.integer(seed)) }
  
  ## ---- Veri hazırlığı ----
  x_mat <- as.matrix(x); storage.mode(x_mat) <- "double"
  class_names <- levels(y)
  n_classes   <- table(y)
  n_needed    <- round((max(n_classes) - n_classes) * ovRate)
  p  <- ncol(x_mat)
  h  <- as.integer(hidden_dim)
  bs <- as.integer(batch_size)
  device <- torch$device(if (torch$cuda$is_available()) "cuda" else "cpu")
  to_t   <- function(m) torch$from_numpy(np_array(m, dtype = "float32"))$to(device)
  
  ## ---- Aşama 1: SMOTE repertuvarı (R tarafında) ----
  smote_repertoire <- function(x_min, n_gen, k) {
    n <- nrow(x_min); k <- min(as.integer(k), n - 1L)
    nn_idx <- FNN::get.knnx(x_min, x_min, k = k + 1L)$nn.index[, -1, drop = FALSE]
    base <- sample.int(n, n_gen, replace = TRUE)
    nbr  <- nn_idx[cbind(base, sample.int(ncol(nn_idx), n_gen, replace = TRUE))]
    x_min[base, , drop = FALSE] + runif(n_gen) * (x_min[nbr, , drop = FALSE] - x_min[base, , drop = FALSE])
  }
  
  ## ---- Aşama 2: GAN ile iyileştirme ----
  train_smotified <- function(x_min_np, n_gen) {
    x_real  <- to_t(x_min_np)
    n_real  <- nrow(x_min_np)
    u_smote <- to_t(smote_repertoire(x_min_np, n_gen, k))
    n_u     <- as.integer(n_gen)
    
    gen_body <- nn$Sequential(
      nn$Linear(as.integer(p), h),     nn$BatchNorm1d(h),     nn$ReLU(),
      nn$Linear(h, as.integer(2L * h)), nn$BatchNorm1d(as.integer(2L * h)), nn$ReLU(),
      nn$Linear(as.integer(2L * h), as.integer(p)))$to(device)
    G <- function(xs) gen_body(xs)$add(xs)                 # artık mantık: SMOTE örneğini iyileştirir
    
    D <- nn$Sequential(
      nn$Linear(as.integer(p), as.integer(2L * h)), nn$LeakyReLU(0.2),
      nn$Linear(as.integer(2L * h), h),             nn$LeakyReLU(0.2),
      nn$Linear(h, 1L), nn$Sigmoid())$to(device)
    
    betas   <- tuple(0.5, 0.999)
    optim_G <- torch$optim$Adam(gen_body$parameters(), lr = lr, betas = betas)
    optim_D <- torch$optim$Adam(D$parameters(),        lr = lr, betas = betas)
    criterion <- nn$BCELoss()
    
    ones  <- torch$ones(bs, device = device)
    zeros <- torch$zeros(bs, device = device)
    
    gen_body$train(); D$train()
    for (epoch in seq_len(epochs)) {
      # Her epoch'ta tek batch, yerine koymalı örnekleme (orijinaldeki gibi)
      real_batch  <- x_real$index_select(0L, torch$randint(0L, as.integer(n_real), tuple(bs)))
      smote_batch <- u_smote$index_select(0L, torch$randint(0L, n_u, tuple(bs)))
      
      # Ayırıcı
      optim_D$zero_grad()
      fake_batch <- G(smote_batch)$detach()
      loss_d <- criterion(D(real_batch)$view(-1L), ones)$add(
        criterion(D(fake_batch)$view(-1L), zeros))
      loss_d$backward()
      optim_D$step()
      
      # Üretici
      optim_G$zero_grad()
      loss_g <- criterion(D(G(smote_batch))$view(-1L), ones)
      loss_g$backward()
      optim_G$step()
    }
    
    # SMOTE repertuvarının son iyileştirmesi
    gen_body$eval()
    with(torch$no_grad(), {
      G(u_smote)$cpu()$numpy()
    })
  }
  
  ## ---- R tarafı: sınıf sınıf üretim ----
  x_syn_list <- list(); y_syn_list <- list()
  for (cls in class_names) {
    needed <- as.integer(n_needed[cls])
    x_cls  <- x_mat[y == cls, , drop = FALSE]
    if (needed <= 0L || nrow(x_cls) < 2L) next             # SMOTE için en az 2 gözlem gerekir
    
    x_syn_np <- as.matrix(train_smotified(x_cls, needed))
    colnames(x_syn_np) <- colnames(x_mat)
    x_syn_list[[cls]] <- x_syn_np
    y_syn_list[[cls]] <- factor(rep(cls, nrow(x_syn_np)), levels = levels(y))
  }
  
  if (length(x_syn_list) == 0L) {
    x_syn <- matrix(numeric(0), nrow = 0, ncol = p, dimnames = list(NULL, colnames(x_mat)))
    y_syn <- factor(character(0), levels = levels(y))
  } else {
    x_syn <- do.call(rbind, unname(x_syn_list))
    y_syn <- do.call(c, unname(y_syn_list))
  }
  
  list(x_new = rbind(x_mat, x_syn), y_new = c(y, y_syn),
       x_syn = x_syn, y_syn = y_syn)
}