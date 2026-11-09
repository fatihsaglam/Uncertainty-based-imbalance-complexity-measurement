## =====================================================================
## analiz_real.R
## run_real.R sonuclarindan (results/real_gains.csv): VERI SETLERI ARASINDA
## "yeniden ornekleme kazanci" ile complexity olculeri arasindaki Spearman korelasyonu.
##  - Siniflayici bazinda (her siniflayicida veri setleri uzerinden), + "Ort" (siniflayicilarin ortalamasi)
##    ve Ort icin veri seti bootstrap'i ile %95 guven araligi.
##  - Standardizasyon: Spearman sira tabanli oldugu icin birimler onemsiz; siniflayicilar birlestirilirken
##    her siniflayici kendi icinde siralanir (ortalama korelasyon).
##  - Tavan etkisi: KONTROL = TRUE ise NORES performansi (sira) etkisi cikarilarak KISMI korelasyon da verilir.
## Kullanim: projenin kok klasorunde  source("analiz_real.R")
## Sadece base R.
## =====================================================================

METRIC <- "GMEAN"                       # "GMEAN", "MCC", "ROCAUC", "P4"
RESAMP <- c("SMOTE", "ROS")             # kazanci hangi yeniden ornekleyiciden
GAINS  <- "results/real_gains.csv"
B      <- 500                           # bootstrap tekrari (guven araligi icin)
SEED   <- 1
options(width = 220)

EU_OL  <- c("EUh_min", "EUh_diff", "EUs_min", "EUs_diff", "EUsb_min", "EUsb_diff")   # h: 5 komsu sert, s: 15 komsu duz oy, sb: 15 komsu agirlikli oy
AU_OL  <- c("AUh_bal", "AUs_bal", "AUsb_bal")
RIV    <- c("IR", "UNSAFE_MIN", "RARE_OUT", "N3_bal", "N1", "N2", "N3", "N4", "LSC", "F1", "F1v", "F2", "F3", "F4", "C2", "ID")
CLF    <- c("CART", "KNN", "RF", "GBM", "LOGREG", "NAIVEBAYES", "SVMRADIAL")

g <- read.csv(GAINS, stringsAsFactors = FALSE)
CLF <- intersect(CLF, unique(g$classifier))
cat("Veri seti:", length(unique(g$dataset)), " | siniflayici:", length(CLF), " | metrik:", METRIC, "\n")
ol_var <- function(cols) intersect(cols, names(g))

## ----------------------------- YARDIMCILAR ---------------------------
## Bir siniflayici icin, veri seti indeksleri ds (tekrarli olabilir) uzerinden korelasyon.
## ctrl: kontrol degiskeni adi (NORES sutunu) ya da NULL
kor <- function(d, cn, gn, ctrl = NULL) {
  x <- d[[cn]]; y <- d[[gn]]; ok <- !is.na(x) & !is.na(y)
  if (!is.null(ctrl)) ok <- ok & !is.na(d[[ctrl]])
  x <- x[ok]; y <- y[ok]
  if (length(unique(x)) < 3 || length(unique(y)) < 3) return(NA_real_)
  rx <- rank(x); ry <- rank(y)
  if (!is.null(ctrl)) { rc <- rank(d[[ctrl]][ok]); rx <- resid(lm(rx ~ rc)); ry <- resid(lm(ry ~ rc)) }
  suppressWarnings(cor(rx, ry))
}
## Tum siniflayicilar icin matris (olcut x siniflayici) ve ortalama
kor_mat <- function(dl, cols, gn, ctrl = NULL) {
  m <- sapply(CLF, function(k) sapply(cols, function(cn) kor(dl[[k]], cn, gn, ctrl)))
  if (is.null(dim(m))) m <- matrix(m, nrow = length(cols), dimnames = list(cols, CLF))
  m
}
## Veri seti bootstrap'i ile Ort icin %95 GA (ayni veri seti ornegi tum siniflayicilarda)
ort_ga <- function(g, cols, gn, ctrl = NULL) {
  set.seed(SEED); ds <- unique(g$dataset)
  bs <- replicate(B, {
    s <- sample(ds, length(ds), replace = TRUE)
    dl <- lapply(CLF, function(k) { d <- g[g$classifier == k, ]; d[match(s, d$dataset), ] })
    names(dl) <- CLF
    rowMeans(kor_mat(dl, cols, gn, ctrl), na.rm = TRUE)
  })
  if (is.null(dim(bs))) bs <- matrix(bs, nrow = length(cols))
  t(apply(bs, 1, quantile, probs = c(0.025, 0.975), na.rm = TRUE))
}
tablo <- function(cols, gn, ctrl = NULL) {
  cols <- ol_var(cols)
  dl <- lapply(CLF, function(k) g[g$classifier == k, ]); names(dl) <- CLF
  m <- kor_mat(dl, cols, gn, ctrl)
  ga <- ort_ga(g, cols, gn, ctrl)
  out <- cbind(round(m, 2), Ort = round(rowMeans(m, na.rm = TRUE), 2), GA_alt = round(ga[, 1], 2), GA_ust = round(ga[, 2], 2))
  rownames(out) <- cols; out
}

## ----------------------------- TABLOLAR ------------------------------
for (rs in RESAMP) {
  gn   <- paste0("gain_", rs, "_", METRIC)
  ctrl <- paste0(METRIC, "_NORES")
  cat("\n=====================================================================\n")
  cat(" KAZANC:", gn, "  (veri setleri arasi Spearman; GA = veri seti bootstrap %95)\n")
  cat("=====================================================================\n")
  cat("\n## EU olculeri ve rakipler\n");            print(tablo(c(EU_OL, RIV), gn))
  cat("\n## AU olculeri ve rakipler\n");            print(tablo(c(AU_OL, RIV), gn))
  cat("\n## KISMI korelasyon (", ctrl, " sirasi kontrol edilir): EU ve AU\n", sep = ""); print(tablo(c(EU_OL, AU_OL, "IR", "UNSAFE_MIN", "N1", "F1v"), gn, ctrl))
}

## Siniflayici bazinda NORES performansi ve kazanc ozeti (veri seti ortalamasi)
cat("\n## Ortalama performans ve kazanc (veri setleri uzerinden)\n")
ozet <- sapply(CLF, function(k) { d <- g[g$classifier == k, ]
  c(NORES = mean(d[[paste0(METRIC, "_NORES")]]), SMOTE = mean(d[[paste0(METRIC, "_SMOTE")]]), ROS = mean(d[[paste0(METRIC, "_ROS")]]),
    "kazanc>0 (SMOTE)" = mean(d[[paste0("gain_SMOTE_", METRIC)]] > 0)) })
print(round(t(ozet), 3))
