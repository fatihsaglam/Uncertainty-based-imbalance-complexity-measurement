library(reticulate)

resamplerCls_AWGAN <- function(x, y, lr = 0.0001, batch_size = 64L, epochs = 100L,
                               hidden_dim = 32L, noise_dim = 16L, lambda_1 = 10,
                               ovRate = 1, seed = NULL, ...) {
  
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
  
  ## ---- Gradyan cezası (WGAN-GP) ----
  gradient_penalty <- function(real, fake, D, center = 1, pw = 2) {
    eta  <- torch$rand(real$size(0L), 1L, device = device)$expand_as(real)
    interp <- eta$mul(real)$add(eta$neg()$add(1)$mul(fake))$requires_grad_(TRUE)
    out  <- D(interp)
    grads <- torch$autograd$grad(outputs = out, inputs = interp,
                                 grad_outputs = torch$ones_like(out),
                                 create_graph = TRUE, retain_graph = TRUE)[[1]]
    grads$norm(p = 2, dim = 1L)$sub(center)$pow(pw)$mean()
  }
  
  ## ---- Tek sınıf için eğit ve üret ----
  train_and_generate <- function(x_cls, n_gen) {
    xr    <- to_t(x_cls)
    n_cls <- nrow(x_cls)
    
    gen <- nn$Sequential(
      nn$Linear(nz, h), nn$BatchNorm1d(h), nn$Mish(),
      nn$Linear(h, h),  nn$BatchNorm1d(h), nn$Mish(),
      nn$Linear(h, p))$to(device)
    D <- nn$Sequential(
      nn$Linear(p, h), nn$Mish(),
      nn$Linear(h, h), nn$Mish(),
      nn$Linear(h, 1L))$to(device)
    G <- function(z, x_real) torch$relu(gen(z)$add(x_real))   # gürültüden gelen delta + gerçek çıpa
    
    optim_G <- torch$optim$AdamW(gen$parameters(), lr = lr)
    optim_D <- torch$optim$AdamW(D$parameters(),   lr = lr)
    
    gen$train(); D$train()
    for (epoch in seq_len(epochs)) {
      perm <- torch$randperm(as.integer(n_cls))
      for (start in seq(0L, n_cls - 1L, by = bs)) {
        len <- min(bs, n_cls - start)
        if (len < 2L) next                                   # BatchNorm tek örnekle çalışmaz
        real_batch <- xr$index_select(0L, perm$narrow(0L, as.integer(start), as.integer(len)))
        
        # Ayırıcı
        optim_D$zero_grad()
        z <- torch$randn(as.integer(len), nz, device = device)
        fake_batch <- G(z, real_batch)$detach()
        d_loss <- D(real_batch)$mean()$neg()$add(D(fake_batch)$mean())$add(
          gradient_penalty(real_batch, fake_batch, D)$div(lambda_1))
        d_loss$backward()
        optim_D$step()
        
        # Üretici
        optim_G$zero_grad()
        z <- torch$randn(as.integer(len), nz, device = device)
        g_loss <- D(G(z, real_batch))$mean()$neg()
        g_loss$backward()
        optim_G$step()
      }
    }
    
    gen$eval()
    with(torch$no_grad(), {
      s_idx <- torch$randint(0L, as.integer(n_cls), tuple(as.integer(n_gen)))
      z     <- torch$randn(as.integer(n_gen), nz, device = device)
      G(z, xr$index_select(0L, s_idx))$cpu()$numpy()
    })
  }
  
  ## ---- R tarafı: sınıf sınıf üretim ----
  x_syn_list <- list(); y_syn_list <- list()
  for (cls in class_names) {
    needed <- as.integer(n_needed[cls])
    x_cls  <- x_mat[y == cls, , drop = FALSE]
    if (needed <= 0L || nrow(x_cls) == 0L) next
    
    x_syn_np <- as.matrix(train_and_generate(x_cls, needed))
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