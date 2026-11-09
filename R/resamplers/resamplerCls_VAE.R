library(reticulate)

resamplerCls_VAE <- function(x, y, lr = 0.001, batch_size = 64L, epochs = 500L,
                             hidden_dim = 32L, latent_dim = 4L, ovRate = 1,
                             seed = NULL, ...) {

  nnf   <- nn$functional
  
  if (!is.null(seed)) { set.seed(seed); torch$manual_seed(as.integer(seed)) }
  
  ## ---- Veri hazırlığı ----
  x_mat <- as.matrix(x); storage.mode(x_mat) <- "double"
  class_names <- levels(y)
  n_classes   <- table(y)
  n_needed    <- round((max(n_classes) - n_classes) * ovRate)
  p    <- as.integer(ncol(x_mat))
  h    <- as.integer(hidden_dim)
  h2   <- as.integer(hidden_dim %/% 2L)                    # orijinaldeki hidden_dim // 2
  lat  <- as.integer(latent_dim)
  bs   <- as.integer(batch_size)
  device <- torch$device(if (torch$cuda$is_available()) "cuda" else "cpu")
  to_t   <- function(m) torch$from_numpy(np_array(m, dtype = "float32"))$to(device)
  
  ## ---- Tek sınıf için eğit ve üret ----
  train_vae <- function(x_main, n_gen) {
    xt <- to_t(x_main)
    n  <- nrow(x_main)
    
    encoder <- nn$Sequential(nn$Linear(p, h), nn$BatchNorm1d(h), nn$ReLU(),
                             nn$Linear(h, h2), nn$ReLU())
    fc_mu     <- nn$Linear(h2, lat)
    fc_logvar <- nn$Linear(h2, lat)
    decoder <- nn$Sequential(nn$Linear(lat, h2), nn$ReLU(),
                             nn$Linear(h2, h), nn$BatchNorm1d(h), nn$ReLU(),
                             nn$Linear(h, p))
    # ModuleList yalnızca optimizer'a ve train()/eval()'e tüm parçaları vermek için
    model <- nn$ModuleList(list(encoder, fc_mu, fc_logvar, decoder))$to(device)
    opt   <- torch$optim$Adam(model$parameters(), lr = lr)
    
    forward <- function(xb) {
      hh     <- encoder(xb)
      mu     <- fc_mu(hh)
      logvar <- fc_logvar(hh)
      std    <- torch$exp(logvar$mul(0.5))
      z      <- mu$add(torch$randn_like(std)$mul(std))     # reparametrizasyon
      list(recon = decoder(z)$add(xb), mu = mu, logvar = logvar)   # artık dönüş: delta + x
    }
    
    vae_loss <- function(recon, xb, mu, logvar) {
      mse <- nnf$mse_loss(recon, xb, reduction = "sum")
      kld <- logvar$add(1)$sub(mu$pow(2))$sub(logvar$exp())$sum()$mul(-0.5)
      mse$add(kld)
    }
    
    model$train()
    for (epoch in seq_len(epochs)) {
      perm <- torch$randperm(as.integer(n))
      for (start in seq(0L, n - 1L, by = bs)) {
        len <- min(bs, n - start)
        if (len < 2L) next                                   # BatchNorm tek örnekle çalışmaz
        batch <- xt$index_select(0L, perm$narrow(0L, as.integer(start), as.integer(len)))
        
        opt$zero_grad()
        o    <- forward(batch)
        loss <- vae_loss(o$recon, batch, o$mu, o$logvar)
        loss$backward()
        opt$step()
      }
    }
    
    # Üretim: z ~ N(0, I) -> decoder, gerçek örnekler çıpa olarak eklenir
    model$eval()
    with(torch$no_grad(), {
      idx_seeds <- torch$randint(0L, as.integer(n), tuple(as.integer(n_gen)))
      x_seeds   <- xt$index_select(0L, idx_seeds)
      z         <- torch$randn(as.integer(n_gen), lat, device = device)
      decoder(z)$add(x_seeds)$cpu()$numpy()
    })
  }
  
  ## ---- R tarafı: sınıf sınıf üretim ----
  x_syn_list <- list(); y_syn_list <- list()
  for (cl in class_names) {
    needed <- as.integer(n_needed[cl])
    x_cls  <- x_mat[y == cl, , drop = FALSE]
    if (needed <= 0L || nrow(x_cls) < 2L) next               # BatchNorm için en az 2 gözlem gerekir
    
    x_gen <- as.matrix(train_vae(x_cls, needed))
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


# set.seed(1)
# x <- matrix(rnorm(300 * 4), 300, 4)
# y <- factor(c(rep("maj", 270), rep("min", 30)), levels = c("maj", "min"))
# system.time(r <- resamplerCls_VAE(x, y, epochs = 20L, seed = 1))
# table(r$y_new); dim(r$x_syn)