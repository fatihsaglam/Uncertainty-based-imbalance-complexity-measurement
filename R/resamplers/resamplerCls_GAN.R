library(reticulate)

resamplerCls_GAN <- function(x, y, lr = 0.001, batch_size = 64L, epochs = 100L,
                             hidden_dim = 16L, noise_dim = 16L, ovRate = 1,
                             seed = NULL, ...) {
  
  if (!is.null(seed)) { set.seed(seed); torch$manual_seed(as.integer(seed)) }
  
  ## ---- Veri hazırlığı ----
  x_mat <- as.matrix(x); storage.mode(x_mat) <- "double"
  class_names <- levels(y)
  n_classes   <- table(y)
  n_needed    <- round((max(n_classes) - n_classes) * ovRate)
  p   <- ncol(x_mat)
  h   <- as.integer(hidden_dim)
  nz  <- as.integer(noise_dim)
  bs  <- as.integer(batch_size)
  device <- torch$device(if (torch$cuda$is_available()) "cuda" else "cpu")
  to_t   <- function(m) torch$from_numpy(np_array(m, dtype = "float32"))$to(device)
  
  ## ---- Tek sınıf için eğit ve üret ----
  train_gan_residual <- function(x_main, n_gen) {
    xt <- to_t(x_main)
    n  <- nrow(x_main)
    
    netG_body <- nn$Sequential(
      nn$Linear(nz, h), nn$BatchNorm1d(h), nn$LeakyReLU(0.2),
      nn$Linear(h, h),  nn$BatchNorm1d(h), nn$LeakyReLU(0.2),
      nn$Linear(h, as.integer(p)))$to(device)
    netG <- function(z, x_real) netG_body(z)$add(x_real)       # artık mantık: gerçek noktadan delta öğrenir
    
    netD <- nn$Sequential(
      nn$Linear(as.integer(p), h), nn$LeakyReLU(0.2),
      nn$Linear(h, h),             nn$LeakyReLU(0.2),
      nn$Linear(h, 1L))$to(device)
    
    betas  <- tuple(0.5, 0.999)
    optimG <- torch$optim$Adam(netG_body$parameters(), lr = lr, betas = betas)
    optimD <- torch$optim$Adam(netD$parameters(),      lr = lr, betas = betas)
    criterion <- nn$BCEWithLogitsLoss()
    
    netG_body$train(); netD$train()
    for (epoch in seq_len(epochs)) {
      perm <- torch$randperm(as.integer(n))
      for (start in seq(0L, n - 1L, by = bs)) {
        len <- min(bs, n - start)
        if (len < 2L) next                                    # BatchNorm tek örnekle çalışmaz
        real_batch  <- xt$index_select(0L, perm$narrow(0L, as.integer(start), as.integer(len)))
        real_labels <- torch$ones(as.integer(len), 1L, device = device)
        fake_labels <- torch$zeros(as.integer(len), 1L, device = device)
        
        # 1) Ayırıcı
        optimD$zero_grad()
        z <- torch$randn(as.integer(len), nz, device = device)
        fake_batch <- netG(z, real_batch)                     # gerçek çıpa verilir
        loss_D <- criterion(netD(real_batch), real_labels)$add(
          criterion(netD(fake_batch$detach()), fake_labels))$div(2)
        loss_D$backward()
        optimD$step()
        
        # 2) Üretici (aynı fake_batch üzerinden)
        optimG$zero_grad()
        loss_G <- criterion(netD(fake_batch), real_labels)
        loss_G$backward()
        optimG$step()
      }
    }
    
    # Son üretim: rastgele gerçek noktalar + öğrenilmiş delta
    netG_body$eval()
    with(torch$no_grad(), {
      idx_seeds <- torch$randint(0L, as.integer(n), tuple(as.integer(n_gen)))
      x_seeds   <- xt$index_select(0L, idx_seeds)
      z_final   <- torch$randn(as.integer(n_gen), nz, device = device)
      netG(z_final, x_seeds)$cpu()$numpy()
    })
  }
  
  ## ---- R tarafı: sınıf sınıf üretim ----
  x_syn_list <- list(); y_syn_list <- list()
  for (cl in class_names) {
    needed <- as.integer(n_needed[cl])
    x_cls  <- x_mat[y == cl, , drop = FALSE]
    if (needed <= 0L || nrow(x_cls) == 0L) next
    
    x_gen <- as.matrix(train_gan_residual(x_cls, needed))
    colnames(x_gen) <- colnames(x_mat)
    x_syn_list[[cl]] <- x_gen
    y_syn_list[[cl]] <- factor(rep(cl, needed), levels = levels(y))
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