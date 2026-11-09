## =====================================================================
## compare_eu_tests.R
## "none" etiketini veren testi karsilastir: Welch t  vs  beta LRT  (ayni alpha'larda)
## ve test GUCUNDEN bagimsiz bir bakis icin ham EU farki / SE de kaydedilir.
## Gercek RF/bootstrap EU degerleri kullanilir (sentetik degil).
## Calistir:  source("global.R")   (euBetaLRT artik complexity_proposed.R icinde; eu_beta_test.R gerekmiyor)  ->  source("compare_eu_tests.R")
## Cikti: results/compare_eu_tests.rds + .csv ; ekrana IR bazinda "none-disi" orani
## =====================================================================
FAMILIES <- c("sep","ovl","sep_scat","disjunct","noise")
ENGINES  <- c("glmmTMB")            # + "glmmTMB" istersen; kurulu degilse acik hata verir
CLFS     <- c("RF", "KNN", "CART")          # istedigin sinifayicilari ekle (non-RF yavas: M=30 bootstrap)
IR_GRID  <- c(2, 5, 10, 25, 50)
N_TOTAL  <- 1500; P_DIM <- 2; R_REP <- 5; SEED <- 2026
ALPHAS   <- c(0.05, 0.10, 0.20)
OUT_DIR  <- "results"; dir.create(OUT_DIR, showWarnings = FALSE)

jobs <- expand.grid(family = FAMILIES, ir = IR_GRID, rep = seq_len(R_REP), clf = CLFS,
                    stringsAsFactors = FALSE)
res <- vector("list", nrow(jobs))
for (k in seq_len(nrow(jobs))) {
  j <- jobs[k, ]
  set.seed(SEED + k)
  n1 <- max(5L, as.integer(round(N_TOTAL / (1 + j$ir)))); n0 <- N_TOTAL - n1
  d  <- dataGeneratorCls_train(j$family, n0 = n0, n1 = n1, P = P_DIM)       # d$y: 1 = azinlik
  X  <- d[, setdiff(names(d), "y"), drop = FALSE]; y <- as.integer(d$y)
  u  <- tryCatch(complexityCls_uncertainty(X, y, clf = j$clf),
                 error = function(e) stop(sprintf("[%s | %s | IR %g | rep %d] %s",
                                                  j$family, j$clf, j$ir, j$rep, conditionMessage(e)), call. = FALSE))
  e1 <- u$eu[u$ok & u$y == 1]; e0 <- u$eu[u$ok & u$y == 0]
  tt <- t.test(e1, e0); b <- euBetaLRT(e1, e0)                 # b: paketsiz (el yazimi) LRT
  pk <- vapply(ENGINES, function(en) euBetaLRTpkg(e1, e0, en)$p, numeric(1))   # hazir paketler
  res[[k]] <- data.frame(j, n1 = length(e1), n0 = length(e0),
                         EU_min = mean(e1), EU_maj = mean(e0), EU_diff_raw = mean(e1) - mean(e0),
                         se_welch = unname(tt$stderr), p_welch = tt$p.value,
                         p_beta_manual = b$p, beta_conv = b$conv, t(setNames(pk, paste0("p_", ENGINES))), row.names = NULL)
  message(sprintf("%d / %d", k, nrow(jobs)))
}
res <- do.call(rbind, res)
saveRDS(res, file.path(OUT_DIR, "compare_eu_tests.rds"))
write.csv(res, file.path(OUT_DIR, "compare_eu_tests.csv"), row.names = FALSE)

## IR x sinifayici: "none-disi" (anlamli) orani, iki test, farkli alpha
for (a in ALPHAS) {
  cat(sprintf("\n== alpha = %.2f : anlamli (none-disi) oran ==\n", a))
  print(round(with(res, tapply(p_welch < a, list(clf, ir), mean)), 2)); cat("-- beta (manuel) --\n")
  print(round(with(res, tapply(p_beta_manual < a, list(clf, ir), mean)), 2))
  for (en in ENGINES) { cat("--", en, "--\n"); print(round(with(res, tapply(res[[paste0("p_", en)]] < a, list(clf, ir), mean)), 2)) }
}