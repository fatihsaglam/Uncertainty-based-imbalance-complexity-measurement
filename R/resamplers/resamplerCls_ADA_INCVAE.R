library(reticulate)

resamplerCls_ADA_INCVAE <- function(x, y, lr = 1e-4, batch_size = 64L, epochs = 500L,
                                    hidden_dim = 32L, latent_dim = 4L, ovRate = 1,
                                    seed = NULL, ...) {

  nnf   <- nn$functional
  
  if (!is.null(seed)) { set.seed(seed); torch$manual_seed(as.integer(seed)) }
  
  ## ---- Veri hazırlığı ----
  x_mat <- as.matrix(x); storage.mode(x_mat) <- "double"
  class_names <- levels(y)
  x_min <- x_mat[y == class_names[2], , drop = FALSE]
  x_maj <- x_mat[y == class_names[1], , drop = FALSE]
  
  n_classes <- table(y)
  n_needed  <- round((max(n_classes) - n_classes) * ovRate)
  n_gen     <- as.integer(n_needed[2])
  
  # Üretilecek örnek yoksa eğitime hiç girmeden boş sonuç döndür
  if (n_gen < 1L) {
    x_gen <- matrix(numeric(0), nrow = 0, ncol = ncol(x_mat), dimnames = list(NULL, colnames(x_mat)))
    y_syn <- factor(character(0), levels = levels(y))
    return(list(x_new = x_mat, y_new = y, x_syn = x_gen, y_syn = y_syn))
  }
  
  h <- as.integer(hidden_dim); lat <- as.integer(latent_dim); bs <- as.integer(batch_size)
  p <- ncol(x_mat); n_min <- nrow(x_min)
  
  device <- torch$device(if (torch$cuda$is_available()) "cuda" else "cpu")
  to_t   <- function(m) torch$from_numpy(np_array(m, dtype = "float32"))$to(device)
  xt_min <- to_t(x_min); xt_maj <- to_t(x_maj)
  
  ## ---- Uyarlanabilir ağırlıklar: çoğunluğa yakın (sınır) azınlık noktaları daha ağır ----
  d_min   <- torch$amin(torch$cdist(xt_min, xt_maj), 1L)      # her azınlık noktasının en yakın çoğunluğa uzaklığı
  weights <- d_min$add(1e-5)$reciprocal()
  weights <- weights$div(weights$mean())
  
  ## ---- Model: encoder, mu / logvar katmanları, decoder ----
  encoder   <- nn$Sequential(nn$Linear(p, h), nn$BatchNorm1d(h), nn$ReLU())
  fc_mu     <- nn$Linear(h, lat)
  fc_logvar <- nn$Linear(h, lat)
  decoder   <- nn$Sequential(nn$Linear(lat, h), nn$ReLU(), nn$Linear(h, p))
  # ModuleList yalnızca optimizer'a ve train()/eval()'e tüm parçaları vermek için
  model <- nn$ModuleList(list(encoder, fc_mu, fc_logvar, decoder))$to(device)
  opt   <- torch$optim$Adam(model$parameters(), lr = lr)
  
  forward <- function(xb) {
    hh     <- encoder(xb)
    mu     <- fc_mu(hh)
    logvar <- fc_logvar(hh)
    std    <- torch$exp(logvar$mul(0.5))
    z      <- mu$add(torch$randn_like(std)$mul(std))     # reparametrizasyon
    list(recon = decoder(z)$add(xb), mu = mu, logvar = logvar)   # delta + x
  }
  
  ## ---- Eğitim ----
  model$train()
  for (epoch in seq_len(epochs)) {
    perm <- torch$randperm(n_min)
    for (start in seq(0L, n_min - 1L, by = bs)) {
      len <- min(bs, n_min - start)
      if (len < 2L) next                                  # BatchNorm tek örnekle çalışmaz
      idx   <- perm$narrow(0L, as.integer(start), as.integer(len))
      batch <- xt_min$index_select(0L, idx)
      bw    <- weights$index_select(0L, idx)
      
      opt$zero_grad()
      o <- forward(batch)
      
      mse_loss <- bw$mul(nnf$mse_loss(o$recon, batch, reduction = "none")$sum(dim = 1L))$mean()
      kld_loss <- o$logvar$add(1)$sub(o$mu$pow(2))$sub(o$logvar$exp())$mean()$mul(-0.5)
      
      loss <- mse_loss$add(kld_loss)
      loss$backward()
      opt$step()
    }
  }
  
  ## ---- Üretim ----
  model$eval()
  x_gen <- with(torch$no_grad(), {
    seeds <- torch$randint(0L, n_min, list(n_gen))
    z     <- torch$randn(n_gen, lat, device = device)
    decoder(z)$add(xt_min$index_select(0L, seeds))$cpu()$numpy()
  })
  x_gen <- as.matrix(x_gen)
  colnames(x_gen) <- colnames(x_mat)
  
  y_syn <- factor(rep(class_names[2], nrow(x_gen)), levels = levels(y))
  list(x_new = rbind(x_mat, x_gen), y_new = c(y, y_syn),
       x_syn = x_gen, y_syn = y_syn)
}