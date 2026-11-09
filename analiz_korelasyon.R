## =====================================================================
## analiz_korelasyon.R
## run.R sonuclarindan (results/sim_cxgain_gains.csv) ozet analiz:
##  1) DENGESIZLIK tarafi: performans dususu ile olcut arasinda, sinif-ici (siniflayici x aile
##     ortalamasi cikarilmis rank'larla) Spearman, siniflayici gruplari bazinda
##  2) IR'nin USTUNE ek R2 (IR modelde; IR dogrusal ve faktor olarak)
##  3) GURULTU tarafi: IR = 1, yanit 1 - GMEAN, aileler arasi
##  4) AYRISIK veri: ayrisik vs diger aile ayirma gucu (IR eslestirilmis AUC) ve olcut ~ IR (0 olmali)
## Kullanim: projenin kok klasorunde  source("analiz_korelasyon.R")
## Sadece base R.
##
## Olcut adlari:
##   EUh_min, EUh_diff, AUh_bal : 5 komsu, SERT bolme (komsularin cogunlugu karsi sinif -> gurultu; hata tabanli)
##   EUs_min, EUs_diff, AUs_bal : 15 komsu, YUMUSAK, DUZ oy        (komsu sonsali q1 = azinlik payi)
##   EUsb_min, EUsb_diff, AUsb_bal : 15 komsu, YUMUSAK, AGIRLIKLI oy (her komsu oyu 1/n_sinif ile agirliklanir; sanki siniflar esit buyuklukte)
##   C1inv = IR; digerleri literatur rakipleri
## =====================================================================

METRIC <- "GMEAN"                        # performans dususu icin metrik: "GMEAN","MCC","ROCAUC","P4" ya da "ORT"
GAINS  <- "results/sim_cxgain_gains.csv"
FAM_IMB   <- c("ovl", "disjunct", "noise")                       # dengesizlik tarafinda aileler
FAM_NOISE <- c("sep", "ovl", "sep_scat", "disjunct", "noise")    # gurultu tarafinda (IR = 1) aileler
FAM_SEP   <- "sep"                                               # ayrisik aile
FAM_OTHER <- c("ovl", "disjunct", "noise")                       # AUC'de karsilastirilan diger aileler

GRUP <- c(CART = "sert", KNN = "sert", RF = "topluluk", GBM = "topluluk",
          LOGREG = "yuksek-sapma", NAIVEBAYES = "yuksek-sapma", SVMRADIAL = "cekirdek")
GRUP_SIRA <- c("sert", "topluluk", "yuksek-sapma", "cekirdek")

EU_OL  <- c("EUh_min", "EUh_diff", "EUs_min", "EUs_diff", "EUsb_min", "EUsb_diff")   # h: 5 komsu sert, s: 15 komsu duz oy, sb: 15 komsu agirlikli oy
AU_OL  <- c("AUh_bal", "AUs_bal", "AUsb_bal")
RIV_IMB <- c("C1inv", "UNSAFE_MIN", "RARE_OUT", "N3_bal", "F1v")    # C1inv = IR
RIV_NOI <- c("F1", "F1v", "N1", "N3", "N4", "UNSAFE_MIN", "RARE_OUT", "N3_bal")

## ----------------------------- VERI ----------------------------------
g <- read.csv(GAINS, stringsAsFactors = FALSE)
g$grp <- unname(GRUP[g$classifier])

dusus <- function(m) {                    # IR = 1 tabani (siniflayici x aile) - satirin NORES degeri
  col <- paste0(m, "_NORES")
  taban <- ave(ifelse(g$IR == 1, g[[col]], NA), g$classifier, g$family, FUN = function(v) mean(v, na.rm = TRUE))
  taban - g[[col]]
}
g$drop <- if (METRIC == "ORT") rowMeans(cbind(dusus("MCC"), dusus("GMEAN"), dusus("ROCAUC"))) else dusus(METRIC)
g$hata <- 1 - g$GMEAN_NORES               # gurultu tarafi yaniti (IR = 1)
ol_var <- function(cols) intersect(cols, names(g))

## ----------------------------- YARDIMCILAR ---------------------------
demean <- function(v, hucre) v - ave(v, hucre, FUN = function(z) mean(z, na.rm = TRUE))

## 1) Grup icinde: tum satirlarin rank'i, hucre (siniflayici x aile) ortalamasi cikarilir, tek korelasyon.
pooled_within <- function(d, cols) {
  cols <- ol_var(cols)
  res <- sapply(cols, function(cn) {
    out <- sapply(GRUP_SIRA, function(k) {
      dk <- d[d$grp == k & !is.na(d[[cn]]), ]
      hucre <- interaction(dk$classifier, dk$family, drop = TRUE)
      x <- demean(rank(dk[[cn]]), hucre); y <- demean(rank(dk$drop), hucre)
      suppressWarnings(cor(x, y))
    })
    c(out, Ort = mean(out, na.rm = TRUE))
  })
  round(t(res), 2)
}

## 2) IR'nin ustune ek R2: yanit ve aciklayicilar rank, hucre ortalamasi cikarilmis; model: drop ~ IR terimleri (+ olcut).
## ir_model: "dogrusal" (IR), "log" (log IR) ya da "faktor" (IR duzeylerinin her biri icin kukla)
ek_r2 <- function(d, cols, ir_model = "faktor") {
  cols <- ol_var(cols)
  res <- sapply(GRUP_SIRA, function(k) {
    dk <- d[d$grp == k, ]
    hucre <- interaction(dk$classifier, dk$family, drop = TRUE)
    y <- demean(rank(dk$drop), hucre)
    Z <- switch(ir_model,
      dogrusal = cbind(dk$IR),
      log      = cbind(log(dk$IR)),
      faktor   = model.matrix(~ factor(IR) - 1, dk)[, -1, drop = FALSE])
    Z <- apply(Z, 2, demean, hucre = hucre)
    r2 <- function(M) { f <- lm.fit(cbind(1, M), y); 1 - sum(f$residuals^2) / sum((y - mean(y))^2) }
    base <- r2(Z)
    c("R2_IR" = base, sapply(cols, function(cn) {
      x <- demean(rank(dk[[cn]]), hucre); ok <- !is.na(x)
      if (any(!ok)) return(NA_real_)
      r2(cbind(Z, x)) - base
    }))
  })
  round(res, 3)
}

## 3) Mann-Whitney AUC: P(x_a > x_b)
auc <- function(a, b) {
  a <- a[!is.na(a)]; b <- b[!is.na(b)]
  r <- rank(c(a, b)); (sum(r[seq_along(a)]) - length(a) * (length(a) + 1) / 2) / (length(a) * length(b))
}

## ----------------------------- 1) DENGESIZLIK ------------------------
imb <- g[g$IR > 1 & g$family %in% FAM_IMB, ]
cat("\n================ METRIK:", METRIC, "================\n")
cat("\n## DENGESIZLIK TARAFI: dusus (IR>1; ", paste(FAM_IMB, collapse = "+"), "; sinif-ici), Spearman\n", sep = "")
print(pooled_within(imb, c(EU_OL, RIV_IMB)))

## ----------------------------- 2) EK R2 -----------------------------
for (irm in c("dogrusal", "faktor")) {
  cat("\n## IR'nin USTUNE ek R2 (IR modelde: ", irm, ")\n", sep = "")
  print(ek_r2(imb, c(EU_OL, setdiff(RIV_IMB, "C1inv")), ir_model = irm))
}

## ----------------------------- 3) GURULTU TARAFI ---------------------
no1 <- g[g$IR == 1 & g$family %in% FAM_NOISE, ]
cat("\n## GURULTU TARAFI: IR = 1, yanit 1 - GMEAN, aileler arasi, Spearman\n")
res <- sapply(ol_var(c(AU_OL, RIV_NOI)), function(cn) {
  out <- sapply(GRUP_SIRA, function(k) { dk <- no1[no1$grp == k, ]
    suppressWarnings(cor(dk[[cn]], dk$hata, method = "spearman", use = "complete.obs")) })
  c(out, Ort = mean(out, na.rm = TRUE)) })
print(round(t(res), 2))

## ----------------------------- 4) AYRISIK VERI ----------------------
sep_ol <- ol_var(c(EU_OL, AU_OL, "UNSAFE_MIN"))
cat("\n## AYRISIK vs DIGER aile, IR-eslestirilmis AUC (1 = ayrisikta dusuk; iyi olan 1'e yakin)\n")
r <- sapply(sep_ol, function(cn) sapply(GRUP_SIRA, function(k) {
  mean(sapply(sort(unique(g$IR)), function(ir) {
    a <- g[g$grp == k & g$IR == ir & g$family %in% FAM_OTHER, cn]
    b <- g[g$grp == k & g$IR == ir & g$family %in% FAM_SEP, cn]
    auc(a, b) })) }))
print(round(t(r), 2))

cat("\n## AYRISIK veride olcut ~ IR (Spearman, siniflayici icinde; 0 olmali)\n")
r <- sapply(sep_ol, function(cn) sapply(GRUP_SIRA, function(k) {
  mean(sapply(unique(g$classifier[g$grp == k]), function(cl) { dk <- g[g$classifier == cl & g$family %in% FAM_SEP, ]
    suppressWarnings(cor(dk[[cn]], dk$IR, method = "spearman", use = "complete.obs")) }), na.rm = TRUE) }))
print(round(t(r), 2))

cat("\n## AYRISIK veride IR'ye gore ortalama (sifira yakin olmali)\n")
print(round(aggregate(g[g$family %in% FAM_SEP, sep_ol], list(IR = g$IR[g$family %in% FAM_SEP]), mean), 3))
