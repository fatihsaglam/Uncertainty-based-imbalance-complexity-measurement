## =====================================================
## dataGenerators.R -- simülasyon veri üreticileri (P değişkenli)
## Sınıf yapısı ilk 2 değişkende; kalan P-2 değişken iki sınıfta da aynı dağılımdan gelen gürültüdür.
## Etiketler: y = 1 azınlık, y = 0 çoğunluk.
## Bu dosya yalnızca tanım içerir, başka dosyaya bağımlı değildir.
## =====================================================

## ---------- Aile tanımları (izotropik Gauss karışımları, ilk 2 boyut) ----------
DATAGEN_SPECS <- list(
  sep      = list(desc = "ayrışık, eşit saçılım",
                  maj = list(c = rbind(c(0, 0)), s = 1),   min = list(c = rbind(c(6, 0)), s = 1)),
  ovl      = list(desc = "kesişen, eşit saçılım",
                  maj = list(c = rbind(c(0, 0)), s = 1),   min = list(c = rbind(c(2, 0)), s = 1)),
  scat_maj = list(desc = "aynı merkez, çoğunluk saçılı",
                  maj = list(c = rbind(c(0, 0)), s = 3),   min = list(c = rbind(c(0, 0)), s = 0.6)),
  sep_scat = list(desc = "ayrışık, çoğunluk saçılı",
                  maj = list(c = rbind(c(0, 0)), s = 2),   min = list(c = rbind(c(9, 0)), s = 0.5)),
  disjunct = list(desc = "azınlık 3 ayrı kümede",
                  maj = list(c = rbind(c(0, 0)), s = 1.5),
                  min = list(c = rbind(c(4, 0), c(-2, 3.46), c(-2, -3.46)), s = 0.8)),
  noise    = list(desc = "ayrışık + etiket gürültüsü",
                  maj = list(c = rbind(c(0, 0)), s = 1),   min = list(c = rbind(c(6, 0)), s = 1))
)
DATAGEN_FAMILIES <- names(DATAGEN_SPECS)       # "noise" ailesi etiket gürültüsü içerir

.dataGeneratorCls_checkFamily <- function(fam) {
  if (!is.character(fam) || length(fam) != 1L || !fam %in% DATAGEN_FAMILIES)
    stop(sprintf("Data generator family '%s' not recognized. Valid: %s",
                 paste(fam, collapse = ","), paste(DATAGEN_FAMILIES, collapse = ", ")), call. = FALSE)
  invisible(TRUE)
}

dataGeneratorCls_featureNames <- function(P) paste0("x", seq_len(P))

## Etiket gürültüsü: yalnızca "noise" ailesinde. Azınlığın gözlenen etiketi 0'a döner (olasılık a),
## çoğunluğun 1'e döner (olasılık b = a * n1 / n0), böylece beklenen sınıf sayıları korunur.
dataGeneratorCls_flipRates <- function(fam, n0, n1, noise_eta = 0.15) {
  .dataGeneratorCls_checkFamily(fam)
  a <- if (fam == "noise") noise_eta else 0
  c(a = a, b = a * n1 / n0)
}

## ---------- Tek sınıftan özellik üretimi ----------
dataGeneratorCls_class <- function(n, spec, P = 2, noise_sd = 1) {
  stopifnot(P >= 2)
  comp <- sample.int(nrow(spec$c), n, replace = TRUE)
  core <- spec$c[comp, , drop = FALSE] + matrix(rnorm(2 * n, sd = spec$s), n, 2)
  Xm <- if (P > 2) cbind(core, matrix(rnorm(n * (P - 2), sd = noise_sd), n, P - 2)) else core
  X <- as.data.frame(Xm); names(X) <- dataGeneratorCls_featureNames(P); X
}

## ---------- Etiketli üretim (gizli sınıf cls, gözlenen etiket gürültülü olabilir) ----------
dataGeneratorCls_labeled <- function(fam, cls = c("min", "maj"), m, P = 2, a = 0, b = 0, noise_sd = 1) {
  .dataGeneratorCls_checkFamily(fam)
  cls <- match.arg(cls); f <- DATAGEN_SPECS[[fam]]
  d <- dataGeneratorCls_class(m, if (cls == "min") f$min else f$maj, P, noise_sd)
  lab <- rep(if (cls == "min") 1L else 0L, m)
  flip <- runif(m) < (if (cls == "min") a else b)
  lab[flip] <- 1L - lab[flip]
  d$y <- lab; d
}

## ---------- Eğitim verisi: n0 çoğunluk + n1 azınlık, karıştırılmış ----------
dataGeneratorCls_train <- function(fam, n0, n1, P = 2, noise_eta = 0.15, noise_sd = 1) {
  fr <- dataGeneratorCls_flipRates(fam, n0, n1, noise_eta)
  d <- rbind(dataGeneratorCls_labeled(fam, "maj", n0, P, fr[["a"]], fr[["b"]], noise_sd),
             dataGeneratorCls_labeled(fam, "min", n1, P, fr[["a"]], fr[["b"]], noise_sd))
  d[sample(nrow(d)), ]
}

## ---------- Test verisi: dengeli, temiz (gizli) etiketli ----------
dataGeneratorCls_test <- function(fam, n_per_class, P = 2, noise_sd = 1) {
  .dataGeneratorCls_checkFamily(fam)
  f <- DATAGEN_SPECS[[fam]]
  rbind(cbind(dataGeneratorCls_class(n_per_class, f$maj, P, noise_sd), y = 0L),
        cbind(dataGeneratorCls_class(n_per_class, f$min, P, noise_sd), y = 1L))
}

## ---------- Bayes aleatoriği (analitik; bit) ----------
## Xbal: dengeli gizli karışımdan örnek (dataGeneratorCls_test çıktısındaki özellikler).
## Yalnızca ilk 2 sütun kullanılır, çünkü gürültü değişkenleri olasılığı değiştirmez.
.dataGeneratorCls_ent <- function(p) {
  p <- pmin(pmax(p, 1e-12), 1 - 1e-12)
  -(p * log2(p) + (1 - p) * log2(1 - p))
}

.dataGeneratorCls_dens <- function(X, spec) {
  X <- as.matrix(X)[, 1:2, drop = FALSE]
  d <- sapply(seq_len(nrow(spec$c)), function(j) {
    r2 <- (X[, 1] - spec$c[j, 1])^2 + (X[, 2] - spec$c[j, 2])^2
    exp(-r2 / (2 * spec$s^2)) / (2 * pi * spec$s^2) })
  rowMeans(matrix(d, nrow = nrow(X)))
}

dataGeneratorCls_bayesAU <- function(fam, Xbal, a = 0, b = 0) {
  .dataGeneratorCls_checkFamily(fam)
  f0 <- .dataGeneratorCls_dens(Xbal, DATAGEN_SPECS[[fam]]$maj)
  f1 <- .dataGeneratorCls_dens(Xbal, DATAGEN_SPECS[[fam]]$min)
  p1 <- f1 / (f0 + f1 + 1e-300)                    # dengeli önsel altında gizli sınıf olasılığı
  mean(.dataGeneratorCls_ent(p1 * (1 - a) + (1 - p1) * b))   # gözlenen etiket olasılığı (gürültü dahil)
}

## ---------- Aile üreticisi: registry (zz_registry.R) bunu kullanır ----------
## Dönen fonksiyon: f(n0, n1, P, noise_eta, noise_sd) -> y sütunlu eğitim verisi
dataGeneratorCls_makeFamily <- function(fam) {
  .dataGeneratorCls_checkFamily(fam)
  force(fam)
  function(n0, n1, P = 2L, noise_eta = 0.15, noise_sd = 1) {
    dataGeneratorCls_train(fam, n0, n1, P, noise_eta, noise_sd)
  }
}

## ---------- Aile üreticileri (her aile ayrı bir fonksiyon) ----------
## Ortak imza: f(n0, n1, P = 2L, noise_eta = 0.15, noise_sd = 1) -> y sütunlu eğitim verisi
## noise_eta yalnızca "noise" ailesinde etkilidir, diğerlerinde yok sayılır.
dataGeneratorCls_sep <- function(n0, n1, P = 2L, noise_eta = 0.15, noise_sd = 1)
  dataGeneratorCls_train("sep", n0, n1, P, noise_eta, noise_sd)

dataGeneratorCls_ovl <- function(n0, n1, P = 2L, noise_eta = 0.15, noise_sd = 1)
  dataGeneratorCls_train("ovl", n0, n1, P, noise_eta, noise_sd)

dataGeneratorCls_scat_maj <- function(n0, n1, P = 2L, noise_eta = 0.15, noise_sd = 1)
  dataGeneratorCls_train("scat_maj", n0, n1, P, noise_eta, noise_sd)

dataGeneratorCls_sep_scat <- function(n0, n1, P = 2L, noise_eta = 0.15, noise_sd = 1)
  dataGeneratorCls_train("sep_scat", n0, n1, P, noise_eta, noise_sd)

dataGeneratorCls_disjunct <- function(n0, n1, P = 2L, noise_eta = 0.15, noise_sd = 1)
  dataGeneratorCls_train("disjunct", n0, n1, P, noise_eta, noise_sd)

dataGeneratorCls_noise <- function(n0, n1, P = 2L, noise_eta = 0.15, noise_sd = 1)
  dataGeneratorCls_train("noise", n0, n1, P, noise_eta, noise_sd)

## ---------- Hızlı kontrol ----------
dataGeneratorCls_describe <- function() {
  data.frame(aile = DATAGEN_FAMILIES,
             aciklama = vapply(DATAGEN_SPECS, function(s) s$desc, character(1)),
             row.names = NULL)
}