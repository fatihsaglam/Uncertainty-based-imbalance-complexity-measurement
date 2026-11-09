## =====================================================================
## threshold_analysis.R
## "Yeniden ornekleme (SMOTE) degerli mi?" karar kurali: EUsb_min > esik.
## Gercek veri: results/real_gains.csv (30 veri seti x 7 siniflayici).
## Calistirma (proje kok klasorunden): source("threshold_analysis.R")
## Cikti: manuscript/figures/threshold_table.csv, fig_real_threshold.pdf
## Yontem: etiket = kazanc > DELTA. Esik = Youden J'yi enbuyuten deger.
##   Basari tahmini: veri seti disarda birakma (leave-one-dataset-out) capraz gecerleme.
##   Esik GA'si ve dogruluk GA'si: veri seti bootstrap'i.
## =====================================================================
REAL  <- "results/real_gains.csv"
OUT   <- "manuscript/figures"
DELTA <- 0.02                        # "deger" sayilan en kucuk kazanc
RESAMP <- "SMOTE"
METRICS <- c(GMEAN = "G-mean", MCC = "MCC", ROCAUC = "ROC-AUC", P4 = "P4")
MEAS <- c("EUsb_min", "EUsb_diff", "IR", "UNSAFE_MIN")   # onerilen iki olcut ve iki rakip
B <- 1000; set.seed(1)
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

r <- read.csv(REAL, stringsAsFactors = FALSE)
CLF <- unique(r$classifier)
auc <- function(s, y) { n1 <- sum(y); n0 <- sum(!y); if (n1 == 0 || n0 == 0) return(NA_real_); (sum(rank(s)[y]) - n1 * (n1 + 1) / 2) / (n1 * n0) }
best_thr <- function(s, y) {
  u <- sort(unique(s)); if (length(u) < 2) return(NA_real_)
  cand <- (u[-1] + u[-length(u)]) / 2
  J <- vapply(cand, function(t) mean(s[y] > t) + mean(s[!y] <= t) - 1, numeric(1))
  cand[which.max(J)]
}
lodo <- function(d, m, y) {          # veri seti disarda birakma tahminleri
  pred <- rep(NA, nrow(d))
  for (ds in unique(d$dataset)) { te <- d$dataset == ds; t <- best_thr(d[[m]][!te], y[!te]); pred[te] <- d[[m]][te] > t }
  pred
}
perf <- function(pred, y) c(sens = mean(pred[y]), spec = mean(!pred[!y]), bacc = (mean(pred[y]) + mean(!pred[!y])) / 2)

res <- list()
for (M in names(METRICS)) {
  gn <- paste0("gain_", RESAMP, "_", M); y <- r[[gn]] > DELTA
  for (m in MEAS) {
    s <- r[[m]]; pr <- lodo(r, m, y); p <- perf(pr, y); t <- best_thr(s, y)
    ## veri seti bootstrap'i: esik ve LODO dogrulugu
    ds <- unique(r$dataset); tb <- numeric(B); bb <- numeric(B)
    for (b in seq_len(B)) {
      sm <- sample(ds, length(ds), replace = TRUE)
      ix <- unlist(lapply(sm, function(z) which(r$dataset == z)))
      tb[b] <- best_thr(s[ix], y[ix]); pp <- perf(pr[ix], y[ix]); bb[b] <- pp["bacc"]
    }
    res[[length(res) + 1]] <- data.frame(
      metric = METRICS[[M]], measure = m, share_worth = mean(y), thr = t,
      thr_lo = unname(quantile(tb, .025, na.rm = TRUE)), thr_hi = unname(quantile(tb, .975, na.rm = TRUE)),
      auc = auc(s, y), sens = p["sens"], spec = p["spec"], bacc = p["bacc"],
      bacc_lo = unname(quantile(bb, .025, na.rm = TRUE)), bacc_hi = unname(quantile(bb, .975, na.rm = TRUE)),
      gain_above = mean(r[[gn]][s > t]), gain_below = mean(r[[gn]][s <= t]), n_above = sum(s > t), n_below = sum(s <= t),
      row.names = NULL)
  }
}
res <- do.call(rbind, res)
write.csv(res, file.path(OUT, "threshold_table.csv"), row.names = FALSE)

## Siniflayici bazinda AUC (EUsb_min, her metrik)
ca <- sapply(names(METRICS), function(M) sapply(CLF, function(k) {
  d <- r[r$classifier == k, ]; auc(d$EUsb_min, d[[paste0("gain_", RESAMP, "_", M)]] > DELTA) }))
colnames(ca) <- unname(METRICS); write.csv(round(ca, 3), file.path(OUT, "threshold_auc_by_classifier.csv"))

## Duyarlilik: DELTA ve ROS
sens <- do.call(rbind, lapply(c(0.01, 0.02, 0.05), function(dl) do.call(rbind, lapply(c("SMOTE", "ROS"), function(rs) {
  gn <- paste0("gain_", rs, "_GMEAN"); y <- r[[gn]] > dl; pr <- lodo(r, "EUsb_min", y)
  data.frame(resampler = rs, delta = dl, share_worth = mean(y), thr = best_thr(r$EUsb_min, y), auc = auc(r$EUsb_min, y), bacc = perf(pr, y)["bacc"], row.names = NULL) }))))
write.csv(sens, file.path(OUT, "threshold_sensitivity_gmean.csv"), row.names = FALSE)

## Sekil: kazanc ~ EUsb_min, 4 metrik
pdf(file.path(OUT, "fig_real_threshold.pdf"), width = 6.6, height = 5.6)
par(mfrow = c(2, 2), mar = c(3.8, 4.4, 1.6, 0.6), mgp = c(2.4, 0.8, 0), family = "serif", las = 1, cex.axis = 0.85)
cl <- c(CART = "#000000", KNN = "#E69F00", RF = "#56B4E9", GBM = "#009E73", LOGREG = "#D55E00", NAIVEBAYES = "#CC79A7", SVMRADIAL = "#0072B2")
for (M in names(METRICS)) {
  gn <- paste0("gain_", RESAMP, "_", M); t <- res$thr[res$metric == METRICS[[M]] & res$measure == "EUsb_min"]
  plot(r$EUsb_min, r[[gn]], pch = 19, cex = 0.55, col = adjustcolor(cl[r$classifier], 0.7), xlab = expression(EU[min]^{sb}),
       ylab = paste("Gain in", METRICS[[M]]), main = sprintf("%s (threshold %.2f)", METRICS[[M]], t), font.main = 1, cex.main = 0.95)
  abline(h = DELTA, col = "grey50", lty = 3); abline(v = t, col = "#D55E00", lty = 2, lwd = 1.4)
}
legend("bottomright", legend = c("CART", "k-NN", "RF", "GBM", "LR", "NB", "SVM"), col = cl, pch = 19, bty = "n", cex = 0.65, ncol = 2)
dev.off()
options(width = 200); print(res[, c("metric", "measure", "thr", "thr_lo", "thr_hi", "auc", "sens", "spec", "bacc", "bacc_lo", "bacc_hi", "gain_above", "gain_below", "n_above", "n_below")], digits = 3)
print(round(ca, 2)); print(sens, digits = 3)
