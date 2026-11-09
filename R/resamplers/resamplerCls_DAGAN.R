library(reticulate)

resamplerCls_DAGAN <- function(x, y, lr = 0.0002, batch_size = 64L, epochs = 200L,
                               hidden_dim = 64L, noise_dim = 100L, lambda_mmd = 0.1,
                               ovRate = 1, seed = NULL, ...) {
  
  nnf   <- nn$functional
  
  if (!is.null(seed)) { set.seed(seed); torch$manual_seed(as.integer(seed)) }
  
  ## ---- Veri hazırlığı ----
  x_mat <- as.matrix(x); storage.mode(x_mat) <- "double"
  class_names <- levels(y)
  n_classes   <- table(y)
  major_cls   <- names(which.max(n_classes))
  n_needed    <- round((max(n_classes) - n_classes) * ovRate)
  p   <- ncol(x_mat)
  h   <- as.integer(hidden_dim)
  nz  <- as.integer(noise_dim)
  bs  <- as.integer(batch_size)
  device <- torch$device(if (torch$cuda$is_available()) "cuda" else "cpu")
  to_t   <- function(m) torch$from_numpy(np_array(m, dtype = "float32"))$to(device)
  
  ## ---- MMD: çok çekirdekli Gauss ----
  gaussian_kernel <- function(source, target, kernel_mul = 2.0, kernel_num = 5L) {
    total <- torch$cat(list(source, target), dim = 0L)
    N <- as.integer(total$size(0L)); d <- as.integer(total$size(1L))
    total0 <- total$unsqueeze(0L)$expand(N, N, d)
    total1 <- total$unsqueeze(1L)$expand(N, N, d)
    L2 <- total0$sub(total1)$pow(2)$sum(dim = 2L)
    bandwidth <- L2$detach()$sum()$div(as.numeric(N)^2 - N)
    bandwidth <- bandwidth$div(kernel_mul^(kernel_num %/% 2L))
    kernels <- lapply(0:(kernel_num - 1L), function(i)
      torch$exp(L2$neg()$div(bandwidth$mul(kernel_mul^i))))
    Reduce(function(a, b) a$add(b), kernels)
  }
  
  mmd_loss <- function(source, target) {
    ns <- as.integer(source$size(0L)); nt <- as.integer(target$size(0L))
    k  <- gaussian_kernel(source, target)
    ss <- k$narrow(0L, 0L, ns)$narrow(1L, 0L, ns)$mean()
    tt <- k$narrow(0L, ns, nt)$narrow(1L, ns, nt)$mean()
    st <- k$narrow(0L, 0L, ns)$narrow(1L, ns, nt)$mean()
    ss$add(tt)$sub(st$mul(2))
  }
  
  ## ---- Bir hedef sınıf için eğit ve üret ----
  train_dagan <- function(x_source, x_target, n_gen) {
    src <- to_t(x_source); tgt <- to_t(x_target)
    n_src <- nrow(x_source); n_tgt <- nrow(x_target)
    
    gen <- nn$Sequential(
      nn$Linear(nz, h), nn$BatchNorm1d(h), nn$ReLU(),
      nn$Linear(h, h),  nn$BatchNorm1d(h), nn$ReLU(),
      nn$Linear(h, p))$to(device)
    G <- function(z, x_real) gen(z)$add(x_real)            # artık bağlantı; son ReLU yok (negatif değerler için)
    
    features   <- nn$Sequential(nn$Linear(p, h), nn$LeakyReLU(0.2),
                                nn$Linear(h, h), nn$LeakyReLU(0.2))$to(device)
    classifier <- nn$Linear(h, 1L)$to(device)
    D <- function(xx) {
      feat <- features(xx)
      list(validity = torch$sigmoid(classifier(feat)), feat = feat)
    }
    
    betas   <- tuple(0.5, 0.999)
    optim_G <- torch$optim$Adam(gen$parameters(), lr = lr, betas = betas)
    optim_D <- torch$optim$Adam(nn$ModuleList(list(features, classifier))$parameters(),
                                lr = lr, betas = betas)
    
    gen$train()
    for (epoch in seq_len(epochs)) {
      # Her epoch'ta tek batch (orijinaldeki gibi)
      len_s <- min(bs, n_src); len_t <- min(bs, n_tgt)
      if (len_s < 2L || len_t < 2L) next
      real_s <- src$index_select(0L, torch$randperm(as.integer(n_src))$narrow(0L, 0L, as.integer(len_s)))
      real_t <- tgt$index_select(0L, torch$randperm(as.integer(n_tgt))$narrow(0L, 0L, as.integer(len_t)))
      
      # Ayırıcı
      optim_D$zero_grad()
      z <- torch$randn(as.integer(len_t), nz, device = device)
      fake_t <- G(z, real_t)$detach()
      d_real <- D(real_t)$validity
      d_fake <- D(fake_t)$validity
      loss_D <- nnf$binary_cross_entropy(d_real, torch$ones_like(d_real))$add(
        nnf$binary_cross_entropy(d_fake, torch$zeros_like(d_fake)))
      loss_D$backward()
      optim_D$step()
      
      # Üretici: çekişmeli kayıp + MMD (alan karışıklığı)
      optim_G$zero_grad()
      z <- torch$randn(as.integer(len_t), nz, device = device)
      o_t    <- D(G(z, real_t))
      feat_s <- D(real_s)$feat
      loss_adv <- nnf$binary_cross_entropy(o_t$validity, torch$ones_like(o_t$validity))
      loss_mmd <- mmd_loss(feat_s, o_t$feat)
      loss_G   <- loss_adv$add(loss_mmd$mul(lambda_mmd))
      loss_G$backward()
      optim_G$step()
    }
    
    gen$eval()
    with(torch$no_grad(), {
      s_idx <- torch$randint(0L, as.integer(n_tgt), tuple(as.integer(n_gen)))
      z     <- torch$randn(as.integer(n_gen), nz, device = device)
      G(z, tgt$index_select(0L, s_idx))$cpu()$numpy()
    })
  }
  
  ## ---- R tarafı: azınlık sınıf(lar) için üretim ----
  x_syn_list <- list(); y_syn_list <- list()
  for (cls in class_names) {
    if (cls == major_cls || n_needed[cls] <= 0) next
    x_tgt <- x_mat[y == cls, , drop = FALSE]
    if (nrow(x_tgt) == 0L) next
    
    x_syn_np <- as.matrix(train_dagan(x_source = x_mat[y == major_cls, , drop = FALSE],
                                      x_target = x_tgt,
                                      n_gen    = as.integer(n_needed[cls])))
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