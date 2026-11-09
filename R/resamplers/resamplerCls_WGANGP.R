library(reticulate)

resamplerCls_WGANGP <- function(x, y, lr = 1e-4, batch_size = 64L, epochs = 1000L,
                                hidden_dim = 64L, noise_dim = 128L, lambda_gp = 10,
                                critic_iters = 5L, ovRate = 1, seed = NULL, ...) {
  
  if (!is.null(seed)) { set.seed(seed); torch$manual_seed(as.integer(seed)) }
  
  ## ---- Veri hazırlığı ----
  x_mat <- as.matrix(x); storage.mode(x_mat) <- "double"
  class_names <- levels(y)
  n_classes   <- table(y)
  n_needed    <- round((max(n_classes) - n_classes) * ovRate)
  p  <- as.integer(ncol(x_mat))
  h  <- as.integer(hidden_dim)
  nz <- as.integer(noise_dim)
  device <- torch$device(if (torch$cuda$is_available()) "cuda" else "cpu")
  to_t   <- function(m) torch$from_numpy(np_array(m, dtype = "float32"))$to(device)
  
  ## ---- Gradyan cezası (WGAN-GP) ----
  calc_gradient_penalty <- function(D, real, fake) {
    alpha  <- torch$rand(real$size(0L), 1L, device = device)$expand_as(real)
    interp <- alpha$mul(real)$add(alpha$neg()$add(1)$mul(fake))$requires_grad_(TRUE)
    d_interp <- D(interp)
    grads <- torch$autograd$grad(outputs = d_interp, inputs = interp,
                                 grad_outputs = torch$ones_like(d_interp),
                                 create_graph = TRUE, retain_graph = TRUE,
                                 only_inputs = TRUE)[[1]]
    grads$norm(p = 2, dim = 1L)$sub(1)$pow(2)$mean()$mul(lambda_gp)
  }
  
  ## ---- Tek sınıf için eğit ve üret ----
  train_wgan_gp <- function(x_main, n_gen) {
    xt <- to_t(x_main)
    n  <- nrow(x_main)
    bs <- as.integer(min(n, as.integer(batch_size)))        # orijinaldeki gibi sınıf boyutuyla sınırlı
    
    netG_body <- nn$Sequential(
      nn$Linear(nz, h), nn$ReLU(),
      nn$Linear(h, as.integer(2L * h)), nn$ReLU(),
      nn$Linear(as.integer(2L * h), p))$to(device)
    netG <- function(z, x_real) netG_body(z)$add(x_real)     # artık bağlantı: gerçek çıpa + delta
    
    netD_body <- nn$Sequential(
      nn$Linear(p, as.integer(2L * h)), nn$LeakyReLU(0.2),
      nn$Linear(as.integer(2L * h), h), nn$LeakyReLU(0.2),
      nn$Linear(h, 1L))$to(device)
    netD <- function(xx) netD_body(xx)$view(-1L)
    
    betas      <- tuple(0.5, 0.9)
    optimizerG <- torch$optim$Adam(netG_body$parameters(), lr = lr, betas = betas)
    optimizerD <- torch$optim$Adam(netD_body$parameters(), lr = lr, betas = betas)
    
    netG_body$train(); netD_body$train()
    for (epoch in seq_len(epochs)) {
      
      # Eleştirmen güncellemeleri
      for (ci in seq_len(critic_iters)) {
        idx <- torch$randperm(as.integer(n))$narrow(0L, 0L, bs)
        real_batch <- xt$index_select(0L, idx)
        b_size <- as.integer(real_batch$size(0L))
        if (b_size < 2L) next
        
        optimizerD$zero_grad()
        z <- torch$randn(b_size, nz, device = device)
        fake_batch <- netG(z, real_batch)$detach()           # D adımında G'ye gradyan gerekmez
        
        loss_D <- netD(fake_batch)$mean()$sub(netD(real_batch)$mean())   # D(fake) - D(real)
        gp     <- calc_gradient_penalty(netD, real_batch$detach(), fake_batch)
        loss_D$add(gp)$backward()
        optimizerD$step()
      }
      
      # Üretici güncellemesi
      optimizerG$zero_grad()
      z <- torch$randn(bs, nz, device = device)
      idx_anchor <- torch$randperm(as.integer(n))$narrow(0L, 0L, bs)
      loss_G <- netD(netG(z, xt$index_select(0L, idx_anchor)))$mean()$neg()
      loss_G$backward()
      optimizerG$step()
    }
    
    netG_body$eval()
    with(torch$no_grad(), {
      idx_seeds <- torch$randint(0L, as.integer(n), tuple(as.integer(n_gen)))
      z_final   <- torch$randn(as.integer(n_gen), nz, device = device)
      netG(z_final, xt$index_select(0L, idx_seeds))$cpu()$numpy()
    })
  }
  
  ## ---- R tarafı: sınıf sınıf üretim ----
  x_syn_list <- list(); y_syn_list <- list()
  for (cl in class_names) {
    needed <- as.integer(n_needed[cl])
    x_cls  <- x_mat[y == cl, , drop = FALSE]
    if (needed <= 0L || nrow(x_cls) == 0L) next
    
    x_gen <- as.matrix(train_wgan_gp(x_cls, needed))
    colnames(x_gen) <- colnames(x_mat)
    x_syn_list[[cl]] <- x_gen
    y_syn_list[[cl]] <- factor(rep(cl, nrow(x_gen)), levels = levels(y))
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