## =====================================================================
## make_figures.R
## Makale sekilleri. Proje kok klasorunden calistirin:  source("make_figures.R")
## Girdi : results/sim_cxgain_gains.csv (simulasyon), results/real_gains.csv (gercek veri)
## Cikti : manuscript/figures/fig_*.pdf
## Sadece base R.
## =====================================================================

SIM  <- "results/sim_cxgain_gains.csv"
REAL <- "results/real_gains.csv"
OUT  <- "manuscript/figures"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

CLF   <- c("CART", "KNN", "RF", "GBM", "LOGREG", "NAIVEBAYES", "SVMRADIAL")
CLAB  <- c(CART = "CART", KNN = "k-NN", RF = "RF", GBM = "GBM", LOGREG = "LR", NAIVEBAYES = "NB", SVMRADIAL = "SVM")
FAM_IMB <- c("ovl", "disjunct", "noise")

## Olcut etiketleri (matematiksel gosterim) ve renkler (Okabe-Ito)
MLABS <- list(EUsb_min = expression(EU[min]^{sb}), EUsb_diff = expression(EU[Delta]^{sb}), IR = expression(IR),
              UNSAFE_MIN = expression(UNS), RARE_OUT = expression(RO), N3_bal = expression(N3[bal]))
RIVALS <- c("IR", "UNSAFE_MIN", "RARE_OUT", "N3_bal")                 # rakip adaylari
PROP   <- c("EUsb_min", "EUsb_diff")
COLS   <- c(EUsb_min = "#D55E00", EUsb_diff = "#E69F00", IR = "#000000", UNSAFE_MIN = "#0072B2", RARE_OUT = "#56B4E9", N3_bal = "#009E73")
PCHS   <- c(EUsb_min = 19, EUsb_diff = 17, IR = 15, UNSAFE_MIN = 1, RARE_OUT = 2, N3_bal = 0)
MEAS   <- c(PROP, RIVALS)
## Cizimde: onerilen 2 olcut + ortalamasi en yuksek 2 rakip
top2 <- function(m) names(sort(colMeans(m[, RIVALS]), decreasing = TRUE))[1:2]
draw_corr <- function(m, ylab, ylim) {
  sel <- c(PROP, top2(m))
  par(mar = c(4.2, 4.4, 1.2, 8.2), mgp = c(2.6, 0.8, 0), cex.axis = 0.9, las = 1, family = "serif")
  plot(NA, xlim = c(0.7, length(CLF) + 0.3), ylim = ylim, xaxt = "n", xlab = "", ylab = ylab)
  abline(h = seq(0, 1, 0.2), col = "grey90"); abline(h = 0, col = "grey60"); axis(1, at = seq_along(CLF), labels = CLAB[CLF])
  for (k in sel) {
    lines(seq_along(CLF), m[, k], col = COLS[k], lwd = 1.8, lty = if (k %in% PROP) 1 else 2)
    points(seq_along(CLF), m[, k], col = COLS[k], pch = PCHS[k], cex = 1.2)
  }
  legend("topright", inset = c(-0.36, 0), xpd = TRUE, legend = do.call(c, unname(MLABS[sel])), col = COLS[sel], pch = PCHS[sel],
         lty = ifelse(sel %in% PROP, 1, 2), bty = "n", cex = 0.9)
  invisible(sel)
}
par_ok <- function() par(mar = c(4.2, 4.4, 1.2, 0.8), mgp = c(2.6, 0.8, 0), cex.axis = 0.9, las = 1, family = "serif")

## ----------------------- Simulasyon ----------------------------------
g <- read.csv(SIM, stringsAsFactors = FALSE)
g$IR_col <- g$IR
base <- ave(ifelse(g$IR == 1, g$GMEAN_NORES, NA), g$classifier, g$family, FUN = function(v) mean(v, na.rm = TRUE))
g$drop <- base - g$GMEAN_NORES
g$IR_m <- g$IR

## Siniflayici bazinda "pooled within" Spearman (IR > 1; ovl, disjunct, noise)
pw <- function(d, cn) {
  d <- d[!is.na(d[[cn]]), ]
  cell <- interaction(d$classifier, d$family, drop = TRUE)
  rx <- ave(rank(d[[cn]]), cell, FUN = function(v) v - mean(v))
  ry <- ave(rank(d$drop),  cell, FUN = function(v) v - mean(v))
  suppressWarnings(cor(rx, ry))
}
imb <- g[g$IR > 1 & g$family %in% FAM_IMB, ]
cs <- sapply(MEAS, function(m) sapply(CLF, function(k) pw(imb[imb$classifier == k, ], if (m == "IR") "C1inv" else m)))
## IR icin C1inv ters isaretli olabilir: mutlak degere gore isaretle
if (cs["CART", "IR"] < 0) cs[, "IR"] <- -cs[, "IR"]
write.csv(round(cs, 3), file.path(OUT, "sim_corr_by_classifier.csv"))

pdf(file.path(OUT, "fig_sim_corr.pdf"), width = 6.6, height = 3.6)
sel_sim <- draw_corr(cs, "Spearman correlation with G-mean drop", c(0, 1))
dev.off()

## Ayrisik veri: EU olcusu IR ile
sep <- g[g$family == "sep", ]
IRs <- sort(unique(sep$IR))
cols <- c("#000000", "#D55E00", "#E69F00", "#009E73", "#0072B2", "#CC79A7", "#56B4E9"); names(cols) <- CLF
pdf(file.path(OUT, "fig_sim_separable.pdf"), width = 6.6, height = 3.4)
par_ok(); par(mar = c(4.2, 4.6, 1.2, 5.2))
mm <- sapply(CLF, function(k) tapply(sep$EUsb_min[sep$classifier == k], sep$IR[sep$classifier == k], mean))
plot(NA, xlim = range(IRs), ylim = c(0, max(mm) * 1.05), log = "x", xlab = "Imbalance ratio (IR)",
     ylab = expression("Mean " * EU[min]^{sb} * " on separable data"), xaxt = "n")
axis(1, at = IRs, labels = IRs); abline(h = 0, col = "grey80")
for (k in CLF) { lines(IRs, mm[, k], col = cols[k], lwd = 1.6); points(IRs, mm[, k], col = cols[k], pch = 19, cex = 0.8) }
legend("topleft", legend = CLAB[CLF], col = cols[CLF], pch = 19, lty = 1, bty = "n", cex = 0.8, ncol = 2)
dev.off()

## Dagilim: siralamalar, 4 siniflayici
pdf(file.path(OUT, "fig_sim_scatter.pdf"), width = 6.6, height = 5.6)
par(mfrow = c(2, 2), mar = c(3.8, 4.2, 1.6, 0.6), mgp = c(2.4, 0.8, 0), family = "serif", las = 1, cex.axis = 0.85)
irc <- c("#A6CEE3", "#1F78B4", "#FDBF6F", "#FF7F00", "#E31A1C"); names(irc) <- as.character(sort(unique(imb$IR)))
for (k in c("CART", "RF", "LOGREG", "SVMRADIAL")) {
  d <- imb[imb$classifier == k, ]
  plot(rank(d$EUsb_min) / nrow(d), rank(d$drop) / nrow(d), pch = 16, cex = 0.8, col = adjustcolor(irc[as.character(d$IR)], 0.6),
       xlab = expression("Rank of " * EU[min]^{sb}), ylab = "Rank of G-mean drop", main = CLAB[k], font.main = 1, cex.main = 0.95)
}
legend("bottomright", legend = paste("IR =", names(irc)), col = irc, pch = 16, bty = "n", cex = 0.7)
dev.off()

## ----------------------- Gercek veri ---------------------------------
r <- read.csv(REAL, stringsAsFactors = FALSE)
GN <- "gain_SMOTE_GMEAN"
rc <- sapply(MEAS, function(m) sapply(CLF, function(k) {
  d <- r[r$classifier == k, ]; suppressWarnings(cor(d[[if (m == "IR") "IR" else m]], d[[GN]], method = "spearman", use = "complete.obs"))
}))
write.csv(round(rc, 3), file.path(OUT, "real_corr_by_classifier.csv"))

pdf(file.path(OUT, "fig_real_corr.pdf"), width = 6.6, height = 3.6)
sel_real <- draw_corr(rc, "Spearman correlation with SMOTE gain", c(min(0, min(rc[, c(PROP, top2(rc))])) - 0.05, 1))
dev.off()

pdf(file.path(OUT, "fig_real_scatter.pdf"), width = 6.6, height = 5.6)
par(mfrow = c(2, 2), mar = c(3.8, 4.4, 1.6, 0.6), mgp = c(2.4, 0.8, 0), family = "serif", las = 1, cex.axis = 0.85)
for (k in c("CART", "RF", "LOGREG", "SVMRADIAL")) {
  d <- r[r$classifier == k, ]
  plot(d$EUsb_min, d[[GN]], pch = 19, cex = 0.8, col = "#D55E00", xlab = expression(EU[min]^{sb}), ylab = "Gain in G-mean (SMOTE)",
       main = sprintf("%s  (Spearman %.2f)", CLAB[k], cor(d$EUsb_min, d[[GN]], method = "spearman")), font.main = 1, cex.main = 0.95)
  abline(h = 0, col = "grey70", lty = 2)
}
dev.off()

cat("Sim rakipler:", sel_sim[-(1:2)], " | Gercek rakipler:", sel_real[-(1:2)], "\n")
cat("Bitti. Sekiller:", normalizePath(OUT), "\n")
print(round(cs, 2)); print(round(rc, 2))
