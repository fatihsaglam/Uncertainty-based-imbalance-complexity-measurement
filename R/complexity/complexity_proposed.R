## =====================================================
## complexity_proposed.R  --  ÖNERİLEN ölçüler
## RF entropi (OOB) + üye başına önsel düzeltmesi -> epistemik (EU) ve aleatorik (AU) belirsizlik
## Yön: yüksek = daha karmaşık. Birim: bit.
## Kural: y in {0,1}, 1 = azınlık (complexityCls_context bunu otomatik sağlar).
## Her ölçü: f(X, y, ctx = NULL, ...). ctx$unc varsa RF yeniden eğitilmez.
## =====================================================
suppressPackageStartupMessages(library(ranger))

.complexityCls_entBits <- function(p) {
  p <- pmin(pmax(p, 1e-12), 1 - 1e-12)
  -(p * log2(p) + (1 - p) * log2(1 - p))
}

# Üye başına önsel düzeltmesi: logit kayması (dengeli referansa geçiş)
.complexityCls_shiftLogit <- function(P, shift) {
  P <- pmin(pmax(P, 1e-6), 1 - 1e-6)
  matrix(plogis(qlogis(P) + shift), nrow = nrow(P))
}

## Dengeli kalibrasyon ozeti. p1 = P(azinlik = 1) (onsel duzeltmeli), y in {0,1}. Sinif bazinda hesaplanip iki sinifin ortalamasi alinir.
##   bece    : en-ust-etiket ECE (esit genislikte nbins kutu, guven = max(p1, 1 - p1)), sinif-kosullu, ortalama
##   berr    : dengeli hata = 1 - ortalama duyarlilik
##   ratio   : bece / berr  (sert 0/1 tahminde = 1; iyi kalibre/dereceli olasiliklarda << 1; berr == 0 ise NA)
complexityCls_balCalib <- function(p1, y, nbins = 10L) {
  ok <- is.finite(p1) & !is.na(y); p1 <- p1[ok]; y <- as.integer(y[ok])
  conf <- pmax(p1, 1 - p1); pred <- as.integer(p1 > 0.5); correct <- as.numeric(pred == y)
  bin  <- cut(conf, seq(0.5, 1, length.out = nbins + 1L), include.lowest = TRUE, labels = FALSE)
  per  <- lapply(0:1, function(k) {
    i <- y == k; if (!any(i)) return(c(ece = NA_real_, err = NA_real_))
    e <- sum(vapply(split(which(i), factor(bin[i], levels = seq_len(nbins))), function(ix)
      if (length(ix)) length(ix) / sum(i) * abs(mean(correct[ix]) - mean(conf[ix])) else 0, numeric(1)))
    c(ece = e, err = 1 - mean(correct[i]))
  })
  m <- rowMeans(do.call(cbind, per))
  c(bece = unname(m["ece"]), berr = unname(m["err"]),
    ratio = if (is.finite(m["err"]) && m["err"] > 0) unname(m["ece"] / m["err"]) else NA_real_)
}

## ---------- Ana hesaplama: gözlem bazlı AU / EU (OOB) ----------
## newdata (opsiyonel, egitim verisi OLMAYAN noktalar, ornegin grid): TUM agaclar kullanilir (OOB kisiti yok),
##   yapraktaki in-bag sayimlar + Laplace + onsel kaymasi; sonuc $new = data.frame(au, eu).
## sample_frac: 1 -> bootstrap (replace = TRUE); < 1 -> her agac verinin sample_frac kadarini YERINE KOYMADAN kullanir
##   (replace = FALSE); geri kalani (1 - sample_frac) OOB olur.
complexityCls_rfEntropyUncertainty <- function(X, y, ntree = 500, min_node_size = 1,
                                               laplace_a = 1, prior_correction = TRUE, seed = 1, newdata = NULL, sample_frac = 1) {
  X <- as.data.frame(X); y <- as.integer(y); n <- nrow(X)
  shift <- if (prior_correction) log(sum(y == 0) / sum(y == 1)) else 0
  rf <- ranger(x = X, y = factor(y), num.trees = ntree, min.node.size = min_node_size,
               replace = sample_frac >= 1, sample.fraction = sample_frac,
               keep.inbag = TRUE, seed = seed)
  tn <- predict(rf, X, type = "terminalNodes")$predictions          # n x ntree
  P <- N <- matrix(NA_real_, n, ntree)
  if (!is.null(newdata)) {
    newdata <- as.data.frame(newdata); tn_new <- predict(rf, newdata, type = "terminalNodes")$predictions
    sumH <- sumP <- numeric(nrow(newdata))                          # ağaçlar üzerinde artımlı toplam (bellek dostu)
  }
  for (t in seq_len(ntree)) {
    w   <- rf$inbag.counts[[t]]
    key <- as.character(tn[, t])
    cp  <- tapply(w * (y == 1), key, sum)                           # yaprak başına in-bag sayımlar
    cn  <- tapply(w * (y == 0), key, sum)
    oob <- w == 0                                                   # yalnızca OOB gözlemler
    P[oob, t] <- cp[key[oob]]; N[oob, t] <- cn[key[oob]]
    if (!is.null(newdata)) {
      kn <- as.character(tn_new[, t])
      pt <- .complexityCls_shiftLogit(matrix((cp[kn] + laplace_a) / (cp[kn] + cn[kn] + 2 * laplace_a), ncol = 1), shift)[, 1]
      sumH <- sumH + .complexityCls_entBits(pt); sumP <- sumP + pt
    }
  }
  p_tree <- .complexityCls_shiftLogit((P + laplace_a) / (P + N + 2 * laplace_a), shift)   # üye başına düzeltme
  au <- rowMeans(.complexityCls_entBits(p_tree), na.rm = TRUE)                  # beklenen entropi
  eu <- pmax(.complexityCls_entBits(rowMeans(p_tree, na.rm = TRUE)) - au, 0)    # karşılıklı bilgi
  new <- if (!is.null(newdata)) { au_n <- sumH / ntree; data.frame(au = au_n, eu = pmax(.complexityCls_entBits(sumP / ntree) - au_n, 0)) }
  list(au = au, eu = eu, y = y, ok = is.finite(au) & is.finite(eu),
       pbar = rowMeans(p_tree, na.rm = TRUE),
       n_oob = rowSums(!is.na(p_tree)), shift = shift, new = new,
       calib = complexityCls_balCalib(rowMeans(p_tree, na.rm = TRUE), y),                  # OOB tabanli dengeli kalibrasyon
       args = list(ntree = ntree, min_node_size = min_node_size,
                   laplace_a = laplace_a, prior_correction = prior_correction, sample_frac = sample_frac))
}

## ---------- Sınıflayıcıya uyarlanmış belirsizlik: sınıf-içi (alt)örnekleme + OOB ----------
## sample_frac: her modelin, HER SINIFTAN kullandigi gozlem orani.
##   sample_frac == 1 -> klasik bootstrap (replace = TRUE; ~%63.2 benzersiz gozlem in-bag)
##   sample_frac <  1 -> yerine koymadan alt-ornek (replace = FALSE; en az 2 gozlem); geri kalani OOB (0.3 -> %70 OOB)
## newdata (opsiyonel, egitim verisi OLMAYAN noktalar, ornegin grid): her model newdata'yi da tahmin eder; sonuc $new =
##   data.frame(au, eu) (tum modellerden).
## noise: 0 = kapali. > 0 ise duzlestirilmis bootstrap: her modelin EGITIM verisinin sayisal ozelliklerine N(0, (noise * sd_j)^2)
##   gurultu eklenir (sd_j: ozelligin tum verideki std. sapmasi); tahmin temiz (gurultusuz) veride yapilir. Hiperparametreler degismez.
complexityCls_bootUncertainty <- function(X, y, clf, M = 30L, prior_correction = TRUE,
                                          seed = 1, fit_args = list(), newdata = NULL, sample_frac = 1, noise = 0) {
  stopifnot(noise >= 0, is.numeric(sample_frac), length(sample_frac) == 1L, sample_frac > 0, sample_frac <= 1)
  X <- as.data.frame(X); y <- as.integer(y); n <- nrow(X)
  shift <- if (prior_correction) log(sum(y == 0) / sum(y == 1)) else 0
  set.seed(seed)
  idx_by_class <- split(seq_len(n), y)
  sdv <- vapply(X, function(v) if (is.numeric(v)) sd(v) else NA_real_, numeric(1))
  P <- matrix(NA_real_, n, M); n_fail <- 0L; n_ok <- 0L
  if (!is.null(newdata)) { newdata <- as.data.frame(newdata); Pnew <- matrix(NA_real_, nrow(newdata), M) }
  
  for (m in seq_len(M)) {
    bs  <- unlist(lapply(idx_by_class, function(i) {
      if (sample_frac >= 1) i[sample.int(length(i), replace = TRUE)]                                                    # bootstrap
      else i[sample.int(length(i), size = min(length(i), max(2L, round(sample_frac * length(i)))), replace = FALSE)]    # alt-ornek
    }), use.names = FALSE)
    oob <- setdiff(seq_len(n), bs)
    if (length(oob) == 0L) next
    pn <- NULL
    p <- tryCatch({
      mod <- get_classifier(clf)
      Xb <- X[bs, , drop = FALSE]
      if (noise > 0) for (j in which(!is.na(sdv))) Xb[[j]] <- Xb[[j]] + rnorm(nrow(Xb), 0, noise * sdv[j])   # duzlestirilmis bootstrap
      do.call(mod$fit, c(list(x = Xb, y = factor(y[bs], levels = c(0, 1))), fit_args))
      get1 <- function(d) { pr <- mod$predict(d)$prob; as.numeric(if (!is.null(colnames(pr))) pr[, "1"] else pr[, 2]) }
      if (!is.null(newdata)) pn <- get1(newdata)
      get1(X[oob, , drop = FALSE])
    }, error = function(e) NULL)
    if (is.null(p)) { n_fail <- n_fail + 1L; next }
    if (!is.null(newdata)) Pnew[, m] <- pn
    P[oob, m] <- p; n_ok <- n_ok + 1L
  }
  if (n_ok == 0L && n_fail < M)
    stop("Hicbir modelde OOB gozlem kalmadi (sample_frac = ", sample_frac, "): EU hesaplanamaz. sample_frac < 1 yapin ya da bootstrap icin 1 kullanin.", call. = FALSE)
  if (n_fail == M) stop(sprintf("Classifier '%s' failed in all %d bootstrap fits.", clf, M), call. = FALSE)
  
  p_m <- .complexityCls_shiftLogit(P, shift)                          # üye başına önsel düzeltmesi
  au  <- rowMeans(.complexityCls_entBits(p_m), na.rm = TRUE)
  eu  <- pmax(.complexityCls_entBits(rowMeans(p_m, na.rm = TRUE)) - au, 0)
  new <- NULL
  if (!is.null(newdata)) {
    pg  <- .complexityCls_shiftLogit(Pnew, shift)
    a   <- rowMeans(.complexityCls_entBits(pg), na.rm = TRUE)
    new <- data.frame(au = a, eu = pmax(.complexityCls_entBits(rowMeans(pg, na.rm = TRUE)) - a, 0))
  }
  list(au = au, eu = eu, y = y, ok = is.finite(au) & is.finite(eu),
       pbar = rowMeans(p_m, na.rm = TRUE),                                  # OOB uyelerin ortalama (onsel duzeltmeli) P(y=1|x)
       n_oob = rowSums(!is.na(p_m)), shift = shift, n_fail = n_fail, new = new,
       ## OOB tabanli dengeli kalibrasyon: prior_correction ayarindan BAGIMSIZ olarak onsel-duzeltmeli olasiliklarla (BECE dengeli popülasyonda anlamli)
       calib = complexityCls_balCalib(rowMeans(.complexityCls_shiftLogit(P, log(sum(y == 0) / sum(y == 1))), na.rm = TRUE), y),
       args = list(clf = clf, M = M, sample_frac = sample_frac, noise = noise, prior_correction = prior_correction))
}

## ---------- Dağıtıcı: tüm sınıflayıcılar (RF dahil) bootstrap sarmalayıcıyla ----------
complexityCls_uncertainty <- function(X, y, clf = "RF", M = 30L, seed = 1,
                                      prior_correction = TRUE, fit_args = list(), newdata = NULL, sample_frac = 1, noise = 0, ...) {
  ## RF dahil TUM siniflayicilar ayni bootstrap sarmalayicidan gecer (uye = siniflayicinin kendisi, RF icin tum orman).
  ## Agac duzeyi hesap icin dogrudan complexityCls_rfEntropyUncertainty() cagrilabilir.
  complexityCls_bootUncertainty(X, y, clf, M, prior_correction, seed, fit_args, newdata = newdata, sample_frac = sample_frac, noise = noise)
}

.complexityCls_getUnc <- function(X, y, ctx = NULL, ...) {
  if (!is.null(ctx) && !is.null(ctx$unc)) return(ctx$unc)
  complexityCls_uncertainty(X, y, ...)
}
.complexityCls_clsMean <- function(u, v, k) mean(u[[v]][u$ok & u$y == k])

## ---------- Beta-dağılım tabanlı olabilirlik oranı testi, paketsiz el yazımı (doğrulama / yedek) ----------
## Model (betareg'deki  y ~ sınıf | sınıf  ile aynı fikir): her sınıf için ayrı ortalama (mu) ve ayrı duyarlılık (phi)
##   y_g ~ Beta(mu_g * phi_g, (1 - mu_g) * phi_g);   H0: mu_azınlık = mu_çoğunluk  (phi'ler sınıfa özgü kalır)
## EU in [0, 1) bit ve çok sayıda (yaklaşık) 0 içerir -> Smithson-Verkuilen ile (0,1)'e sıkıştırılır.
## Not: yalnızca "fark var mı" kararı için; EU_min/EU_maj/EU_diff değerleri hâlâ ham sınıf ortalamalarıdır.
.complexityCls_betaNLL <- function(y, mu, phi) -sum(dbeta(y, mu * phi, (1 - mu) * phi, log = TRUE))

.complexityCls_betaFitGroup <- function(y) {                       # tek grup: (mu, phi) MLE
  m <- mean(y); v <- var(y)
  phi0 <- if (is.finite(v) && v > 0) max(m * (1 - m) / v - 1, 1) else 1e3
  o <- optim(c(qlogis(m), log(min(phi0, 1e6))),
             function(p) .complexityCls_betaNLL(y, plogis(p[1]), exp(p[2])),
             method = "L-BFGS-B", lower = c(-20, log(0.1)), upper = c(20, log(1e6)))   # phi sınırlı: sabit grup -> sonsuz olabilirlik olmasın
  list(nll = o$value, mu = plogis(o$par[1]), phi = exp(o$par[2]), conv = o$convergence)
}

## Dönüş: list(p, stat, mu1, mu0, conv). p = NA yalnızca grup < 3 gözlem ise.
euBetaLRT <- function(e1, e0) {
  e1 <- e1[is.finite(e1)]; e0 <- e0[is.finite(e0)]
  n1 <- length(e1); n0 <- length(e0); n <- n1 + n0
  if (n1 < 3 || n0 < 3) return(list(p = NA_real_, stat = NA_real_, mu1 = NA_real_, mu0 = NA_real_, conv = NA_integer_))
  sq <- function(y) (y * (n - 1) + 0.5) / n                         # [0,1] -> (0,1)
  y1 <- sq(e1); y0 <- sq(e0)
  f1 <- .complexityCls_betaFitGroup(y1); f0 <- .complexityCls_betaFitGroup(y0)
  nll1 <- f1$nll + f0$nll                                           # H1: mu'lar serbest
  m <- (sum(y1) + sum(y0)) / n
  o <- optim(c(qlogis(m), log(f1$phi), log(f0$phi)),                # H0: ortak mu, sınıfa özgü phi
             function(p) .complexityCls_betaNLL(y1, plogis(p[1]), exp(p[2])) +
               .complexityCls_betaNLL(y0, plogis(p[1]), exp(p[3])),
             method = "L-BFGS-B", lower = c(-20, log(0.1), log(0.1)), upper = c(20, log(1e6), log(1e6)))
  stat <- max(2 * (o$value - nll1), 0)
  list(p = pchisq(stat, 1, lower.tail = FALSE), stat = stat, mu1 = f1$mu, mu0 = f0$mu,
       conv = max(f1$conv, f0$conv, o$convergence))
}

## ---------- Beta regresyonu: HAZIR PAKETLERLE (betareg / glmmTMB) ----------
## Model: ortalama ~ sınıf, duyarlılık (phi) ~ sınıf  (değişen dağılım).  H0: ortalama sınıftan bağımsız.
## LRT = 2 * (logLik(tam) - logLik(H0)), df = 1.  EU [0,1) -> Smithson-Verkuilen ile (0,1)'e sıkıştırılır.
## engine = "betareg" : betareg::betareg(y ~ sinif | sinif)
## engine = "glmmTMB" : glmmTMB::glmmTMB(y ~ sinif, dispformula = ~ sinif, family = beta_family())
## Not: tek veri setinde küme yok -> rastgele etki anlamsız; glmmTMB burada yalnızca beta GLM olarak kullanılır.
## Dönüş: list(p, stat, mu1, mu0, conv (mantıksal, TRUE = yakınsadı), engine_used, note).
euBetaLRTpkg <- function(e1, e0, engine = c("betareg", "glmmTMB"), on_fail = c("manual", "stop")) {
  engine <- match.arg(engine); on_fail <- match.arg(on_fail)
  if (!requireNamespace(engine, quietly = TRUE))
    stop("'", engine, "' paketi kurulu değil: install.packages(\"", engine, "\")", call. = FALSE)
  e1 <- e1[is.finite(e1)]; e0 <- e0[is.finite(e0)]
  n1 <- length(e1); n0 <- length(e0); n <- n1 + n0
  if (n1 < 3 || n0 < 3)
    return(list(p = NA_real_, stat = NA_real_, mu1 = NA_real_, mu0 = NA_real_, conv = NA,
                engine_used = engine, note = "n < 3"))
  d <- data.frame(y = (c(e1, e0) * (n - 1) + 0.5) / n,
                  sinif = factor(rep(c("min", "maj"), c(n1, n0)), levels = c("maj", "min")))
  
  ## Paket modeli, bir sınıfın EU'su (neredeyse) sabitse (örn. hepsi 0) tanımsızdır: phi -> sonsuz, bilgi matrisi
  ## tekil ("failed to invert the information matrix" / "chol.default(K)"). Böyle durumda ya da paket hata verirse:
  ##   on_fail = "manual": phi'si sınırlı el yazımı LRT'ye (euBetaLRT) DÜŞÜLÜR; gerekçe $note'ta, kullanılan yöntem $engine_used'da.
  ##   on_fail = "stop"  : hata verir.
  fallback <- function(reason) {
    if (on_fail == "stop") stop(engine, ": ", reason, call. = FALSE)
    r <- euBetaLRT(e1, e0)
    r$conv <- !is.na(r$conv) && r$conv == 0                           # mantıksal: TRUE = yakınsadı
    c(r, list(engine_used = "beta_manual", note = paste0(engine, " kullanılamadı: ", reason)))
  }
  sdg <- tapply(d$y, d$sinif, sd)
  if (any(!is.finite(sdg) | sdg < 1e-6))
    return(fallback("sınıflardan birinde EU (neredeyse) sabit, örn. hepsi 0"))
  
  fit <- tryCatch({
    if (engine == "betareg") {
      m1 <- betareg::betareg(y ~ sinif | sinif, data = d)
      m0 <- betareg::betareg(y ~ 1 | sinif,     data = d)
      list(m1 = m1, m0 = m0, mu = tapply(fitted(m1), d$sinif, mean),
           conv = isTRUE(m1$converged) && isTRUE(m0$converged))
    } else {
      m1 <- glmmTMB::glmmTMB(y ~ sinif, dispformula = ~ sinif, family = glmmTMB::beta_family(), data = d)
      m0 <- glmmTMB::glmmTMB(y ~ 1,     dispformula = ~ sinif, family = glmmTMB::beta_family(), data = d)
      list(m1 = m1, m0 = m0, mu = tapply(predict(m1, type = "response"), d$sinif, mean),
           conv = isTRUE(m1$sdr$pdHess) && isTRUE(m0$sdr$pdHess))
    }
  }, error = function(e) e)
  if (inherits(fit, "error")) return(fallback(conditionMessage(fit)))
  
  stat <- max(2 * (as.numeric(logLik(fit$m1)) - as.numeric(logLik(fit$m0))), 0)
  list(p = pchisq(stat, 1, lower.tail = FALSE), stat = stat,
       mu1 = unname(fit$mu["min"]), mu0 = unname(fit$mu["maj"]), conv = fit$conv,
       engine_used = engine, note = "")
}

## Paket sonucundaki düşüşü/yakınsamayı GÖRÜNÜR yapar (sessiz geçmez).
.complexityCls_pkgWarn <- function(r, engine) {
  if (nzchar(r$note) && r$note != "n < 3") warning(r$note, " -> ", r$engine_used, " ile hesaplandı", call. = FALSE)
  else if (!is.na(r$conv) && !r$conv) warning(engine, " yakınsamadı", call. = FALSE)
}

## ---------- Sınıflar arası gözlem bazlı EU farkı anlamlı mı? ----------
## ESKİ (Welch t, gözlemleri bağımsız sayar) -- silinmedi, yorum satırı:
# .complexityCls_euSignificant <- function(u, alpha = 0.05) {
#   e1 <- u$eu[u$ok & u$y == 1]; e0 <- u$eu[u$ok & u$y == 0]
#   if (length(e1) < 2 || length(e0) < 2) return(FALSE)
#   p <- tryCatch(t.test(e1, e0)$p.value, error = function(e) NA_real_)
#   is.finite(p) && p < alpha
# }
## YENİ: test = "betareg" (varsayılan) | "glmmTMB" | "beta_manual" (paketsiz el yazımı LRT) | "welch" (eski).
## Varsayılanı options(complexityCls.euTest = "welch") gibi değiştirmek mümkün.
.complexityCls_euSignificant <- function(u, alpha = 0.05, test = getOption("complexityCls.euTest", "betareg")) {
  test <- match.arg(test, c("betareg", "glmmTMB", "beta_manual", "welch"))
  e1 <- u$eu[u$ok & u$y == 1]; e0 <- u$eu[u$ok & u$y == 0]
  if (length(e1) < 2 || length(e0) < 2) return(FALSE)
  p <- switch(test,
              welch       = t.test(e1, e0)$p.value,
              beta_manual = { r <- euBetaLRT(e1, e0)
              if (!is.na(r$conv) && r$conv != 0) warning("euBetaLRT: optim yakınsamadı (kod ", r$conv, ")", call. = FALSE)
              r$p },
              betareg     = { r <- euBetaLRTpkg(e1, e0, "betareg"); .complexityCls_pkgWarn(r, "betareg"); r$p },
              glmmTMB     = { r <- euBetaLRTpkg(e1, e0, "glmmTMB"); .complexityCls_pkgWarn(r, "glmmTMB"); r$p })
  is.finite(p) && p < alpha
}

# Ortalama EU oranı için log-ölçekte test: H0: log((m1+eps)/(m0+eps)) = 0; delta yöntemiyle z testi
.complexityCls_euLogRatioSig <- function(u, eps = 1e-3, alpha = 0.05) {
  e1 <- u$eu[u$ok & u$y == 1]; e0 <- u$eu[u$ok & u$y == 0]
  n1 <- length(e1); n0 <- length(e0)
  if (n1 < 2 || n0 < 2) return(FALSE)
  m1 <- mean(e1) + eps; m0 <- mean(e0) + eps
  se <- sqrt(var(e1) / (n1 * m1^2) + var(e0) / (n0 * m0^2))
  if (!is.finite(se) || se == 0) return(FALSE)
  2 * pnorm(-abs(log(m1 / m0) / se)) < alpha
}

## ---------- HATA TABANLI ayrisim: komsulukla gurultu / gurultusuz ----------
## Gozlem hatasi e_i = 1 - pbar_i(gercek sinif)  (pbar: OOB uyelerin ortalama, onsel duzeltmeli P(y=1|x)).
## Komsuluk: ctx$nnk (complexityCls_context) -- yoksa ctx$D'den/yeniden hesaplanir.
##   q1_i = komsularin azinlik payi; balanced = TRUE ise her komsunun oyu 1/n_sinif ile agirliklanir (onsel etkisini kaldirir).
.complexityCls_obsErr <- function(u) ifelse(u$y == 1L, 1 - u$pbar, u$pbar)

.complexityCls_nbrPost <- function(X, y, ctx, k, balanced) {
  cx <- if (!is.null(ctx) && !is.null(ctx$nnk)) ctx else complexityCls_context(X, y, knn_k = max(k, 15L))
  nn <- cx$nnk[, seq_len(k), drop = FALSE]; n <- length(y)
  w  <- if (balanced) 1 / as.numeric(table(y)[as.character(y)]) else rep(1, n)
  rowSums(matrix(w[nn], n, k) * matrix(y[nn] == 1L, n, k)) / rowSums(matrix(w[nn], n, k))
}

.complexityCls_errSplit <- function(X, y, ctx, k, balanced, ...) {
  u  <- .complexityCls_getUnc(X, y, ctx, ...)
  q1 <- .complexityCls_nbrPost(X, y, ctx, k, balanced)
  list(y = u$y, e = .complexityCls_obsErr(u), pbar = u$pbar, q1 = q1, same = ifelse(u$y == 1L, q1, 1 - q1),
       ok = u$ok & is.finite(u$pbar))
}

## (1) SERT bolme: 5 komsunun cogunlugu KARSI sinifsa "gurultu". Gurultulu gozlemlerin hatasi -> aleatorik (AUh),
##     gurultusuz gozlemlerin hatasi -> epistemik (EUh).
##     cond = FALSE (varsayilan): sinif bazinda toplam hata iki parcaya BOLUNUR (AUh_c + EUh_c = sinifin ortalama OOB hatasi);
##       yani her grubun hatasi sinifin tum gozlemlerine bolunur ve grubun payi hesaba girer.
##     cond = TRUE: grup ici kosullu ortalama (gurultu payi hesaba girmez; AUh gurultu azken bile yuksek kalir, bos grup: AU 0, EU NA).
.complexityCls_hardParts <- function(s, cond = FALSE) {
  noise <- s$same < 0.5
  m <- function(cl, nz) {
    ic <- s$ok & s$y == cl; i <- ic & noise == nz
    if (cond) { if (any(i)) mean(s$e[i]) else if (nz) 0 else NA_real_ } else sum(s$e[i]) / max(sum(ic), 1L)
  }
  c(au1 = m(1L, TRUE), au0 = m(0L, TRUE), eu1 = m(1L, FALSE), eu0 = m(0L, FALSE))
}
complexityCls_EUh_min  <- function(X, y, ctx = NULL, k = 5L, balanced = FALSE, cond = FALSE, ...)
  unname(.complexityCls_hardParts(.complexityCls_errSplit(X, y, ctx, k, balanced, ...), cond)["eu1"])
complexityCls_EUh_diff <- function(X, y, ctx = NULL, k = 5L, balanced = FALSE, cond = FALSE, ...) {
  p <- .complexityCls_hardParts(.complexityCls_errSplit(X, y, ctx, k, balanced, ...), cond); unname(p["eu1"] - p["eu0"])
}
complexityCls_AUh_bal  <- function(X, y, ctx = NULL, k = 5L, balanced = FALSE, cond = FALSE, ...) {
  p <- .complexityCls_hardParts(.complexityCls_errSplit(X, y, ctx, k, balanced, ...), cond); unname(0.5 * (p["au1"] + p["au0"]))
}

## (2) YUMUSAK: esik yok. Yerel sonsal q1 (komsu oylari; varsayilan duz oy = egitimdeki sinif orani, onsel duzeltmesi kapaliyken pbar ile tutarli) ile modelin olasiligi pbar karsilastirilir.
##     aleatorik_i = min(q1, 1 - q1)  (yerel Bayes hatasi), epistemik_i = |pbar - q1|  (model, yerel sonsaldan ne kadar uzak).
##     Sinif bazinda ortalama.
.complexityCls_softParts <- function(s) {
  a <- pmin(s$q1, 1 - s$q1); d <- abs(s$pbar - s$q1)
  m <- function(v, cl) { i <- s$ok & s$y == cl; if (any(i)) mean(v[i]) else NA_real_ }
  c(au1 = m(a, 1L), au0 = m(a, 0L), eu1 = m(d, 1L), eu0 = m(d, 0L))
}
complexityCls_EUs_min  <- function(X, y, ctx = NULL, k = 15L, balanced = FALSE, ...)
  unname(.complexityCls_softParts(.complexityCls_errSplit(X, y, ctx, k, balanced, ...))["eu1"])
complexityCls_EUs_diff <- function(X, y, ctx = NULL, k = 15L, balanced = FALSE, ...) {
  p <- .complexityCls_softParts(.complexityCls_errSplit(X, y, ctx, k, balanced, ...)); unname(p["eu1"] - p["eu0"])
}
complexityCls_AUs_bal  <- function(X, y, ctx = NULL, k = 15L, balanced = FALSE, ...) {
  p <- .complexityCls_softParts(.complexityCls_errSplit(X, y, ctx, k, balanced, ...)); unname(0.5 * (p["au1"] + p["au0"]))
}

## ---------- Ölçüler ----------
complexityCls_EU_min  <- function(X, y, ctx = NULL, ...) .complexityCls_clsMean(.complexityCls_getUnc(X, y, ctx, ...), "eu", 1)
complexityCls_EU_maj  <- function(X, y, ctx = NULL, ...) .complexityCls_clsMean(.complexityCls_getUnc(X, y, ctx, ...), "eu", 0)
complexityCls_EU_max  <- function(X, y, ctx = NULL, ...) {
  u <- .complexityCls_getUnc(X, y, ctx, ...)
  max(.complexityCls_clsMean(u, "eu", 1), .complexityCls_clsMean(u, "eu", 0))
}
complexityCls_EU_diff <- function(X, y, ctx = NULL, alpha = 0.05, ...) {
  u <- .complexityCls_getUnc(X, y, ctx, ...)
  if (!.complexityCls_euSignificant(u, alpha)) return(0)
  .complexityCls_clsMean(u, "eu", 1) - .complexityCls_clsMean(u, "eu", 0)
}
complexityCls_EU_asym <- function(X, y, ctx = NULL, alpha = 0.05, ...) abs(complexityCls_EU_diff(X, y, ctx, alpha, ...))
complexityCls_EU_ratio <- function(X, y, ctx = NULL, eps = 1e-3, alpha = 0.05, ...) {
  u <- .complexityCls_getUnc(X, y, ctx, ...)
  if (!.complexityCls_euLogRatioSig(u, alpha)) return(1)
  s <- c(.complexityCls_clsMean(u, "eu", 1), .complexityCls_clsMean(u, "eu", 0))
  (max(s) + eps) / (min(s) + eps)
}

complexityCls_AU_min  <- function(X, y, ctx = NULL, ...) .complexityCls_clsMean(.complexityCls_getUnc(X, y, ctx, ...), "au", 1)
complexityCls_AU_maj  <- function(X, y, ctx = NULL, ...) .complexityCls_clsMean(.complexityCls_getUnc(X, y, ctx, ...), "au", 0)
complexityCls_AU_bal  <- function(X, y, ctx = NULL, ...) {
  u <- .complexityCls_getUnc(X, y, ctx, ...)
  0.5 * (.complexityCls_clsMean(u, "au", 1) + .complexityCls_clsMean(u, "au", 0))
}

## ---------- Önerilen ölçülerin listesi ----------
PROPOSED_MEASURES <- list(
  EU_min = complexityCls_EU_min, EU_maj = complexityCls_EU_maj, EU_max = complexityCls_EU_max,
  EU_diff = complexityCls_EU_diff, EU_asym = complexityCls_EU_asym, EU_ratio = complexityCls_EU_ratio,
  AU_min = complexityCls_AU_min, AU_maj = complexityCls_AU_maj, AU_bal = complexityCls_AU_bal,
  EUh_min = complexityCls_EUh_min, EUh_diff = complexityCls_EUh_diff, AUh_bal = complexityCls_AUh_bal,   # hata tabanli, sert bolme
  EUs_min = complexityCls_EUs_min, EUs_diff = complexityCls_EUs_diff, AUs_bal = complexityCls_AUs_bal)   # hata tabanli, yumusak

PROPOSED_INFO <- data.frame(
  name  = names(PROPOSED_MEASURES),
  group = c(rep("EU (dengesizlik)", 6), rep("AU (overlap/gürültü)", 3),
            "EU (hata, sert)", "EU (hata, sert)", "AU (hata, sert)", "EU (hata, yumusak)", "EU (hata, yumusak)", "AU (hata, yumusak)"),
  ref   = "bu çalışma", stringsAsFactors = FALSE)