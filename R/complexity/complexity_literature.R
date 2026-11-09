## =====================================================
## complexity_literature.R  --  LİTERATÜRDEKİ ölçüler
## Tüm ölçüler yönlendirilmiş: yüksek = daha karmaşık / daha dengesiz.
## Hepsi f(X, y, ctx = NULL). İkili sınıflama (y in {0,1}, 1 = azınlık).
## Tanımlar hafızamdan (ECoL, Ho & Basu, Ortigosa-Hernández, Napierała); kullanmadan önce kaynaktan doğrula.
## =====================================================
suppressPackageStartupMessages(library(class))

## ---------- Etiket kodlama: azınlık = 1 ----------
.complexityCls_encodeMinority <- function(y) {
  if (all(as.character(y) %in% c("0", "1"))) {
    y <- as.integer(as.character(y))
    if (sum(y == 1) > sum(y == 0)) y <- 1L - y          # eşitlikte olduğu gibi bırak
    return(y)
  }
  yf <- factor(y); stopifnot(nlevels(yf) == 2)
  tb <- table(yf); as.integer(yf == names(tb)[which.min(tb)])
}

## ---------- Ortak bağlam: mesafe ve en yakın komşu bir kez hesaplanır ----------
complexityCls_context <- function(X, y, scale_type = c("zscore", "minmax"),
                                  with_distances = TRUE, with_proposed = FALSE, proposed_args = list(), knn_k = 15L) {
  scale_type <- match.arg(scale_type)
  X <- as.data.frame(X); y <- .complexityCls_encodeMinority(y); Xm <- as.matrix(X)
  Xs <- if (scale_type == "zscore") scale(Xm) else
    apply(Xm, 2, function(v) { r <- diff(range(v)); if (r == 0) rep(0, length(v)) else (v - min(v)) / r })
  Xs <- as.matrix(Xs); Xs[!is.finite(Xs)] <- 0
  ctx <- list(X = X, Xm = Xm, Xs = Xs, y = y, n = nrow(X), scale_type = scale_type)
  if (with_distances) {
    D <- as.matrix(dist(Xs)); diag(D) <- Inf
    ctx$D <- D; ctx$nn1 <- max.col(-D, ties.method = "first")
    kk <- min(as.integer(knn_k), nrow(D) - 1L)                                   # en yakin kk komsu (yakindan uzaga), n x kk
    ctx$nnk <- matrix(t(apply(D, 1, function(r) order(r)[seq_len(kk)])), nrow = nrow(D))
  }
  if (with_proposed)
    ctx$unc <- do.call(complexityCls_uncertainty, c(list(X = X, y = y), proposed_args))
  ctx
}
.complexityCls_getCtx <- function(X, y, ctx) if (is.null(ctx)) complexityCls_context(X, y) else ctx

## =====================================================
## A) Sınıf dengesizliği ölçüleri (yalnızca y)
## =====================================================
complexityCls_IR <- function(X = NULL, y, ctx = NULL) { tb <- as.numeric(table(y)); max(tb) / min(tb) }

# 1 - ECoL C1 (C1 = sınıf oranlarının normalize entropisi; yüksek = dengeli)
complexityCls_C1inv <- function(X = NULL, y, ctx = NULL) {
  p <- as.numeric(table(y)) / length(y); 1 + sum(p * log(p)) / log(length(p))
}

# ECoL C2: 1 - 1/IR_çoksınıflı ; IR_ç = (K-1)/K * sum(n_k / (n - n_k))
complexityCls_C2 <- function(X = NULL, y, ctx = NULL) {
  tb <- as.numeric(table(y)); n <- sum(tb); K <- length(tb)
  1 - 1 / ((K - 1) / K * sum(tb / (n - tb)))
}

# Imbalance Degree (Öklid mesafesi): ID = d(p, e) / d(iota_m, e) + (m - 1)
complexityCls_ID <- function(X = NULL, y, ctx = NULL) {
  p <- as.numeric(table(y)) / length(y); K <- length(p); e <- rep(1 / K, K)
  m <- sum(p < 1 / K); if (m == 0) return(0)
  iota <- c(rep(0, m), rep(1 / (K - m), K - m))
  d <- function(a, b) sqrt(sum((a - b)^2))
  d(p, e) / d(iota, e) + (m - 1)
}

## =====================================================
## B) Özellik tabanlı overlap (Ho & Basu / ECoL)
## =====================================================
complexityCls_F1 <- function(X, y, ctx = NULL) {         # max Fisher oranı, 1/(1+max r)
  ctx <- .complexityCls_getCtx(X, y, ctx); Xm <- ctx$Xm; y <- ctx$y
  r <- apply(Xm, 2, function(v) {
    a <- v[y == 1]; b <- v[y == 0]; den <- var(a) + var(b)
    if (!is.finite(den) || den == 0) 0 else (mean(a) - mean(b))^2 / den })
  1 / (1 + max(r))
}

complexityCls_F1v <- function(X, y, ctx = NULL) {        # yön vektörlü Fisher
  ctx <- .complexityCls_getCtx(X, y, ctx); Xm <- ctx$Xm; y <- ctx$y
  p1 <- mean(y == 1); p0 <- 1 - p1
  X1 <- Xm[y == 1, , drop = FALSE]; X0 <- Xm[y == 0, , drop = FALSE]
  m1 <- colMeans(X1); m0 <- colMeans(X0); m <- p1 * m1 + p0 * m0
  W <- p1 * cov(X1) + p0 * cov(X0)
  W <- W + diag(1e-8 * mean(diag(W)) + 1e-12, ncol(W))   # tekillik için küçük ridge
  d <- solve(W, m1 - m0)
  num <- p1 * sum(d * (m1 - m))^2 + p0 * sum(d * (m0 - m))^2
  1 / (1 + num / as.numeric(t(d) %*% W %*% d))
}

.complexityCls_featInterval <- function(v, y) c(lo = max(min(v[y == 1]), min(v[y == 0])),
                                                hi = min(max(v[y == 1]), max(v[y == 0])))

complexityCls_F2 <- function(X, y, ctx = NULL) {         # örtüşme hacmi
  ctx <- .complexityCls_getCtx(X, y, ctx); Xm <- ctx$Xm; y <- ctx$y
  prod(sapply(seq_len(ncol(Xm)), function(j) {
    v <- Xm[, j]; iv <- .complexityCls_featInterval(v, y)
    rg <- max(v) - min(v); if (rg == 0) 1 else max(0, iv[["hi"]] - iv[["lo"]]) / rg }))
}

complexityCls_F3 <- function(X, y, ctx = NULL) {         # en iyi özelliğin örtüşme payı
  ctx <- .complexityCls_getCtx(X, y, ctx); Xm <- ctx$Xm; y <- ctx$y
  min(sapply(seq_len(ncol(Xm)), function(j) {
    v <- Xm[, j]; iv <- .complexityCls_featInterval(v, y)
    if (iv[["lo"]] > iv[["hi"]]) 0 else mean(v >= iv[["lo"]] & v <= iv[["hi"]]) }))
}

complexityCls_F4 <- function(X, y, ctx = NULL) {         # kolektif özellik verimliliği
  ctx <- .complexityCls_getCtx(X, y, ctx); Xm <- ctx$Xm; y <- ctx$y; n <- ctx$n
  keep <- rep(TRUE, n); feats <- seq_len(ncol(Xm))
  while (length(feats) > 0 && any(keep)) {
    if (length(unique(y[keep])) < 2) { keep[] <- FALSE; break }
    cnt <- sapply(feats, function(j) {
      iv <- .complexityCls_featInterval(Xm[keep, j], y[keep])
      if (iv[["lo"]] > iv[["hi"]]) 0 else sum(Xm[keep, j] >= iv[["lo"]] & Xm[keep, j] <= iv[["hi"]]) })
    jb <- feats[which.min(cnt)]
    iv <- .complexityCls_featInterval(Xm[keep, jb], y[keep])
    if (iv[["lo"]] > iv[["hi"]]) { keep[] <- FALSE; break }
    keep <- keep & Xm[, jb] >= iv[["lo"]] & Xm[, jb] <= iv[["hi"]]
    feats <- setdiff(feats, jb)
  }
  sum(keep) / n
}

## =====================================================
## C) Komşuluk tabanlı ölçüler (D, nn1 gerekir)
## =====================================================
.complexityCls_mstEdges <- function(D) {                 # Prim, O(n^2)
  n <- nrow(D); in_tree <- rep(FALSE, n); in_tree[1] <- TRUE
  best <- D[1, ]; from <- rep(1L, n); edges <- matrix(0L, n - 1, 2)
  for (k in seq_len(n - 1)) {
    cand <- best; cand[in_tree] <- Inf
    j <- which.min(cand); edges[k, ] <- c(from[j], j); in_tree[j] <- TRUE
    upd <- D[j, ] < best & !in_tree
    best[upd] <- D[j, upd]; from[upd] <- j
  }
  edges
}

complexityCls_N1 <- function(X, y, ctx = NULL) {         # MST'de farklı sınıfa bağlı nokta oranı
  ctx <- .complexityCls_getCtx(X, y, ctx); E <- .complexityCls_mstEdges(ctx$D)
  dc <- ctx$y[E[, 1]] != ctx$y[E[, 2]]
  length(unique(c(E[dc, 1], E[dc, 2]))) / ctx$n
}

complexityCls_N2 <- function(X, y, ctx = NULL) {         # sınıf içi / sınıflar arası mesafe oranı
  ctx <- .complexityCls_getCtx(X, y, ctx); D <- ctx$D; same <- outer(ctx$y, ctx$y, "==")
  Ds <- D; Ds[!same] <- Inf; intra <- apply(Ds, 1, min)
  De <- D; De[same]  <- Inf; extra <- apply(De, 1, min)
  intra[is.infinite(intra)] <- NA
  r <- sum(intra, na.rm = TRUE) / sum(extra); r / (1 + r)
}

complexityCls_N3 <- function(X, y, ctx = NULL) {         # 1NN LOO hatası
  ctx <- .complexityCls_getCtx(X, y, ctx); mean(ctx$y[ctx$nn1] != ctx$y)
}

complexityCls_N3_bal <- function(X, y, ctx = NULL) {     # sınıf dengeli 1NN LOO hatası (varyant)
  ctx <- .complexityCls_getCtx(X, y, ctx); err <- ctx$y[ctx$nn1] != ctx$y
  0.5 * (mean(err[ctx$y == 1]) + mean(err[ctx$y == 0]))
}

complexityCls_N4 <- function(X, y, ctx = NULL, seed = NULL) {   # enterpolasyonlu noktalarda 1NN hatası
  ctx <- .complexityCls_getCtx(X, y, ctx); if (!is.null(seed)) set.seed(seed)
  Xs <- ctx$Xs; y <- ctx$y; Z <- NULL; zy <- integer(0)
  for (k in 0:1) {
    i <- which(y == k); m <- length(i)
    a <- sample(i, m, replace = TRUE); b <- sample(i, m, replace = TRUE); t <- runif(m)
    Z <- rbind(Z, Xs[a, , drop = FALSE] * (1 - t) + Xs[b, , drop = FALSE] * t); zy <- c(zy, rep(k, m))
  }
  pred <- class::knn(Xs, Z, factor(y), k = 1)
  mean(as.integer(as.character(pred)) != zy)
}

complexityCls_LSC <- function(X, y, ctx = NULL) {        # yerel küme kardinalitesi: 1 - sum|LS| / n^2
  ctx <- .complexityCls_getCtx(X, y, ctx); D <- ctx$D; same <- outer(ctx$y, ctx$y, "==")
  De <- D; De[same] <- Inf; enemy <- apply(De, 1, min)
  Ds <- D; Ds[!same] <- Inf
  1 - sum(rowSums(Ds < enemy)) / ctx$n^2
}

## =====================================================
## D) Azınlık yapısı (Napierała & Stefanowski; k = 5)
## =====================================================
.complexityCls_minSameCounts <- function(ctx, k = 5) {
  idx1 <- which(ctx$y == 1)
  nn <- t(apply(ctx$D[idx1, , drop = FALSE], 1, function(r) order(r)[1:k]))
  rowSums(matrix(ctx$y[nn] == 1, nrow = length(idx1)))   # 5 komşudan kaçı azınlık
}
complexityCls_UNSAFE_MIN <- function(X, y, ctx = NULL)   # güvenli olmayan azınlık oranı
  mean(.complexityCls_minSameCounts(.complexityCls_getCtx(X, y, ctx)) <= 3)
complexityCls_RARE_OUT <- function(X, y, ctx = NULL)     # nadir + aykırı azınlık oranı
  mean(.complexityCls_minSameCounts(.complexityCls_getCtx(X, y, ctx)) <= 1)

## =====================================================
## LİTERATÜR ÖLÇÜLERİ LİSTESİ
## =====================================================
LITERATURE_MEASURES <- list(
  IR = complexityCls_IR, C1inv = complexityCls_C1inv, C2 = complexityCls_C2, ID = complexityCls_ID,
  F1 = complexityCls_F1, F1v = complexityCls_F1v, F2 = complexityCls_F2, F3 = complexityCls_F3, F4 = complexityCls_F4,
  N1 = complexityCls_N1, N2 = complexityCls_N2, N3 = complexityCls_N3, N3_bal = complexityCls_N3_bal,
  N4 = complexityCls_N4, LSC = complexityCls_LSC,
  UNSAFE_MIN = complexityCls_UNSAFE_MIN, RARE_OUT = complexityCls_RARE_OUT)

LITERATURE_INFO <- data.frame(
  name  = names(LITERATURE_MEASURES),
  group = c(rep("dengesizlik", 4), rep("overlap (özellik)", 5), rep("komşuluk", 6), rep("azınlık yapısı", 2)),
  ref   = c("sayım", "ECoL C1", "ECoL C2", "Ortigosa-Hernández",
            "Ho & Basu", "Ho & Basu", "Ho & Basu", "Ho & Basu", "Ho & Basu",
            "Ho & Basu", "ECoL", "Ho & Basu", "N3 varyantı", "Ho & Basu", "Leyva ve ark.",
            "Napierała & Stefanowski", "Napierała & Stefanowski"),
  stringsAsFactors = FALSE)