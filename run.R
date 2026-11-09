## =====================================================================
## run.R  (ana simulasyon: complexity vs kazanc)
## Amac: Dengesiz TRAIN verisinde hesaplanan (sadece onerdigimiz) complexity
##       olculerini, "dogru dagilimdan N_ADD veri eklemenin" DENGELI TEST
##       verisindeki performans kazanciyla birlikte toplamak.
##       KORELASYON HESAPLANMIYOR: sadece ham veri toplanir, analiz sonra.
##
## Tasarim
##  - Aileler : sep, ovl, sep_scat, disjunct, noise
##  - IR      : 1, 2, 5, 10, 25, 50  (N_TOTAL sabit)
##  - Yontemler (egitim verisi varyantlari):
##      NORES   : dengesiz train aynen
##      SMOTE   : SMOTE (ovRate = 1)            -- sadece performans toplanir
##      ROS     : ROS   (ovRate = 1)            -- sadece performans toplanir
##      ADD_MAJ : train + dogru dagilimdan N_ADD gercek COGUNLUK ornegi
##      ADD_MIN : train + dogru dagilimdan N_ADD gercek AZINLIK ornegi
##  - Metrik  : ROCAUC, MCC, P4, GMEAN  (dengeli test, N_TEST_PER_CLASS/sinif)
##  - Kazanc  : gain_maj_<M> = M(ADD_MAJ) - M(NORES)
##              gain_min_<M> = M(ADD_MIN) - M(NORES)   (ayni id + siniflayici)
##              SMOTE/ROS performanslari kazanca GIRMEZ, sadece kaydedilir.
##  - Complexity (her siniflayici icin AYRI, dengesiz train uzerinde, hata tabanli):
##      hata e_i = 1 - (OOB ortalama olasiligin gercek sinifi)
##      5 komsu, SERT bolme  : komsularin cogunlugu karsi sinif -> gurultu.
##          AUh_bal  = gurultu grubunun sinif bazli ortalama hatasi (dengeli ort.)   [aleatorik]
##          EUh_min / EUh_diff = gurultu OLMAYAN grubun sinif bazli ortalama hatasi (azinlik / azinlik - cogunluk) [epistemik]
##      15 komsu, YUMUSAK    : a_i = min(q1, 1 - q1), d_i = |pbar_i - q1_i|  (q1: komsu sonsali)
##          AUs_bal, EUs_min, EUs_diff   -> duz oy (her komsu esit)
##          AUsb_bal, EUsb_min, EUsb_diff -> agirlikli oy (komsu oylari sinif buyuklugune gore; sanki siniflar esit buyuklukte)
##    + tum rakip (literatur) olculer (siniflayicidan bagimsiz)
##
##
## Calistirma: projenin kok klasorunde  source("run.R")  (analiz/testler burada yok; sonuclar results/ altina yazilir,
## testEquivalence / testDirection / testCorrelation icin playground dosyalarini kullan).
## Tek dosya: add_cx.R yok. Cikti (results/): checkpoint, sim_cxgain_results.rds (perf, cx, gains),
##   sim_cxgain_gains.csv (genis tablo: meta + tum performanslar + kazanclar + cx),
##   sim_cxgain_perf_long.csv (yontem basina uzun performans tablosu)
## =====================================================================

source("global.R")            # reticulate/torch, library()'ler ve tum dosyalar

suppressPackageStartupMessages({ library(parallel) })

## ----------------------------- AYARLAR -------------------------------
FAMILIES    <- c("sep","ovl","sep_scat","disjunct","noise")
CLASSIFIERS <- c("LOGREG","NAIVEBAYES","KNN","CART","RF","SVMRADIAL","GBM")
METHODS     <- c("NORES","SMOTE","ROS","ADD_MAJ","ADD_MIN")
IR_GRID     <- c(1, 2, 5, 10, 25, 50)
N_TOTAL     <- 1500
N_ADD       <- 300            # eklenen gercek ornek sayisi (ADD_MAJ / ADD_MIN)
N_TEST_PER_CLASS <- 1500      # dengeli test: toplam 3000
P_DIM       <- 2
R_REP       <- 30
N_CORES     <- max(1, detectCores() - 1)    # once 1 ile calistir (hata gorunur); sonra max(1, detectCores() - 1)
SEED        <- 2026
OUT_DIR     <- "results"          # tum cikti dosyalari (rds, csv, ...) buraya
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
CKPT        <- file.path(OUT_DIR, "sim_cxgain_checkpoint.rds")
PRIOR_CORRECTION <- FALSE      # complexity (EU/AU) hesabinda uye basina onsel duzeltmesi: TRUE = logit + log(n0/n1), FALSE = kapali
                               # (sinif-dengeli kalibrasyon sutunlari oob_*/test_* bu ayardan bagimsiz, her zaman duzeltmeli)
## Agirlikli oylu yumusak surum: complexity_proposed.R'deki fonksiyonlarin balanced = TRUE cagrisi
PROPOSED_MEASURES$EUsb_min  <- function(X, y, ctx = NULL, ...) complexityCls_EUs_min(X, y, ctx = ctx, balanced = TRUE, ...)
PROPOSED_MEASURES$EUsb_diff <- function(X, y, ctx = NULL, ...) complexityCls_EUs_diff(X, y, ctx = ctx, balanced = TRUE, ...)
PROPOSED_MEASURES$AUsb_bal  <- function(X, y, ctx = NULL, ...) complexityCls_AUs_bal(X, y, ctx = ctx, balanced = TRUE, ...)
CX_NAMES    <- c("EUh_min","EUh_diff","AUh_bal",         # 5 komsu, sert bolme
                 "EUs_min","EUs_diff","AUs_bal",         # 15 komsu, yumusak, duz oy
                 "EUsb_min","EUsb_diff","AUsb_bal")      # 15 komsu, yumusak, agirlikli oy
## Eski olculeri de istersen: CX_NAMES'e "EU_min","EU_maj","EU_diff","EU_ratio","AU_bal" ekle (PROPOSED_MEASURES'ta duruyor).
LIT_NAMES   <- setdiff(names(LITERATURE_MEASURES), "IR")   # rakip (literatur) olculer; IR zaten gains$IR sutunu
METRICS     <- c("ROCAUC","MCC","P4","GMEAN")
CLASS_COL   <- "Class"
POS         <- "min"          # azinlik = pozitif sinif
NEG         <- "maj"

## Checkpoint imzasi: ayarlar degistiyse eski checkpoint ile karismasin.
## (R_REP imzada yok: artirirsan yeni tekrarlar sona eklenir, eski idler degismez.)
## Kodun kendisini degistirdiysen checkpoint'i elle sil.
SIG <- list(FAMILIES, CLASSIFIERS, METHODS, IR_GRID, N_TOTAL, N_ADD,
            N_TEST_PER_CLASS, P_DIM, SEED, LIT_NAMES, CX_NAMES, PRIOR_CORRECTION)

## ----------------------------- ADAPTORLER ----------------------------
## Veri: dataGenerators.R (dataGeneratorCls_*). Gizli sinif y: 1 = azinlik.
## Burada x1..xP + Class (faktor, levels = c(NEG, POS)) formatina cevriliyor.
.to_class_df <- function(d) {
  d$Class <- factor(ifelse(d$y == 1L, POS, NEG), levels = c(NEG, POS))
  d$y <- NULL
  d
}
.split_n <- function(n_total, ir) {
  n1 <- max(5L, as.integer(round(n_total / (1 + ir))))
  c(n0 = as.integer(n_total - n1), n1 = n1)
}
gen_train <- function(family, n_total, ir, p = P_DIM) {
  n <- .split_n(n_total, ir)
  .to_class_df(dataGeneratorCls_train(family, n0 = n[["n0"]], n1 = n[["n1"]], P = p))
}
gen_test <- function(family, n_per_class, p = P_DIM) {
  .to_class_df(dataGeneratorCls_test(family, n_per_class = n_per_class, P = p))
}
## Egitim dagilimindan (etiket gurultusu dahil, ayni flip oranlariyla) m ek ornek, tek sinifli.
gen_extra <- function(family, cls, m, n_total, ir, p = P_DIM) {
  n  <- .split_n(n_total, ir)
  fr <- dataGeneratorCls_flipRates(family, n[["n0"]], n[["n1"]])
  .to_class_df(dataGeneratorCls_labeled(family, cls, m, P = p, a = fr[["a"]], b = fr[["b"]]))
}

## Resampler: get_resampler(name)$fun(x, y, ...) -> list(x_new, y_new)
## Hiperparametreler sabit (hypGrid taranmiyor): ovRate = 1 (tam dengeleme), k = 5.
RS_ARGS <- list(SMOTE = list(k = 5, ovRate = 1), ROS = list(ovRate = 1))
do_resample <- function(name, train) {
  if (length(unique(table(train[[CLASS_COL]]))) == 1L) return(train)   # zaten dengeli (IR = 1)
  x <- train[, setdiff(names(train), CLASS_COL), drop = FALSE]
  r <- do.call(get_resampler(name)$fun, c(list(x = x, y = train[[CLASS_COL]]), RS_ARGS[[name]]))
  out <- as.data.frame(r$x_new); names(out) <- names(x)
  out[[CLASS_COL]] <- factor(r$y_new, levels = c(NEG, POS))
  out
}

## Yonteme gore egitim verisi varyanti
make_variant <- function(method, train, extra_maj, extra_min) {
  out <- switch(method,
    NORES   = train,
    SMOTE   = , ROS = do_resample(method, train),
    ADD_MAJ = rbind(train, extra_maj),
    ADD_MIN = rbind(train, extra_min),
    stop("bilinmeyen yontem: ", method))
  rownames(out) <- NULL
  out
}

## Siniflayici: R6, $fit(x, y) / $predict(x) -> list(pred, prob); prob kolon adlari = sinif adlari.
## Metrik fonksiyonlari (truth, pred, prob) bekliyor; prob kolonlari faktor level sirasinda (NEG, POS).
fit_predict <- function(clf_name, train, test) {
  xc  <- setdiff(names(train), CLASS_COL)
  clf <- get_classifier(clf_name)
  clf$fit(train[, xc, drop = FALSE], train[[CLASS_COL]])
  pr <- clf$predict(test[, xc, drop = FALSE])
  list(pred = factor(pr$pred, levels = c(NEG, POS)),
       prob = as.matrix(pr$prob)[, c(NEG, POS), drop = FALSE])
}

## Onerilen complexity (sadece EU/AU), siniflayiciya ozgu, DENGESIZ train.
## complexityCls_uncertainty(X, y, clf) gozlem bazli AU/EU'yu bir kez hesaplar (y: 0/1, 1 = azinlik);
## olculer PROPOSED_MEASURES icinden ctx$unc ile ayni hesaptan turetilir (yeniden egitim yok).
## ctx0: complexityCls_context(X, y) (mesafe + komsu indeksleri; veri setine ozgu, siniflayicidan bagimsiz, bir kez hesaplanir).
get_proposed_cx <- function(clf_name, train, ctx0) {
  X <- train[, setdiff(names(train), CLASS_COL), drop = FALSE]
  y <- as.integer(train[[CLASS_COL]] == POS)
  ctx <- ctx0; ctx$unc <- complexityCls_uncertainty(X, y, clf = clf_name, prior_correction = PRIOR_CORRECTION)
  out <- vapply(PROPOSED_MEASURES[CX_NAMES], function(f) as.numeric(f(X, y, ctx = ctx)), numeric(1))
  attr(out, "calib") <- ctx$unc$calib                       # OOB tabanli dengeli kalibrasyon (bece, berr, ratio)
  out
}

## Test tabanli dengeli kalibrasyon (metrik kutuphanesindeki BECE / BECERatio / BACC ile): olasiliklar onsel duzeltmeli
## (egitimdeki sinif oraniyla ayni logit kaymasi), tahmin = duzeltilmis olasiligin argmax'i.
test_calib <- function(pr, truth, train) {
  shift <- log(sum(train[[CLASS_COL]] == NEG) / sum(train[[CLASS_COL]] == POS))
  p1    <- .complexityCls_shiftLogit(matrix(pr$prob[, POS], ncol = 1), shift)[, 1]
  prob2 <- cbind(1 - p1, p1); colnames(prob2) <- c(NEG, POS)
  truth <- factor(as.character(truth), levels = c(NEG, POS))
  pred  <- factor(ifelse(p1 > 0.5, POS, NEG), levels = c(NEG, POS))
  c(bece = metricCls_BECE(truth, prob = prob2), berr = 1 - metricCls_BACC(truth, pred),
    ratio = metricCls_BECERatio(truth, prob = prob2))
}

## Metrikler: mevcut metrik fonksiyonlari (metricCls_all) kullaniliyor, yeniden yazilmadi.
## Hepsi f(truth, pred, prob, ...) ; P4 icin class_pos/class_neg acikca veriliyor
## (dengeli testte which.min(table(truth)) ilk leveli secerdi).
perf_metrics <- function(pr, truth) {
  fs <- get_metrics(METRICS)
  vapply(fs, function(f)
    as.numeric(f(truth = truth, pred = pr$pred, prob = pr$prob,
                 class_pos = POS, class_neg = NEG)), numeric(1))
}

## ----------------------------- TEK KOSU ------------------------------
## HATA BASTIRMA YOK: hata olursa kosu durur, mesaja hangi is/siniflayici/yontem
## oldugu eklenir (hata yutulmaz, sadece baglam eklenip yeniden firlatilir).
with_ctx <- function(label, expr) {
  tryCatch(expr, error = function(e)
    stop(sprintf("[%s] %s", label, conditionMessage(e)), call. = FALSE))
}

run_one <- function(job) {
  set.seed(job$seed)
  lab <- sprintf("id=%d family=%s IR=%s rep=%d", job$id, job$family, job$ir, job$rep)
  train <- with_ctx(paste(lab, "| veri uretimi"), gen_train(job$family, N_TOTAL, job$ir))
  test  <- with_ctx(paste(lab, "| test uretimi"), gen_test(job$family, N_TEST_PER_CLASS))
  ## Ek ornekler siniflayici dongusunden ONCE uretilir (rastgele sayi akisi sabit kalsin)
  extra_maj <- with_ctx(paste(lab, "| ek cogunluk"), gen_extra(job$family, "maj", N_ADD, N_TOTAL, job$ir))
  extra_min <- with_ctx(paste(lab, "| ek azinlik"),  gen_extra(job$family, "min", N_ADD, N_TOTAL, job$ir))

  Xl <- train[, setdiff(names(train), CLASS_COL), drop = FALSE]; yl <- as.integer(train[[CLASS_COL]] == POS)
  ctxl <- with_ctx(paste(lab, "| complexity baglami"), complexityCls_context(Xl, yl))   # RNG kullanmaz
  perf <- list(); cx <- list(); cal <- list()
  for (clf in CLASSIFIERS) {
    cxv <- with_ctx(paste(lab, "| complexity |", clf), get_proposed_cx(clf, train, ctxl))
    cx[[clf]] <- data.frame(id = job$id, family = job$family, IR = job$ir,
                            rep = job$rep, classifier = clf, t(cxv))
    co <- attr(cxv, "calib")
    for (mt in METHODS) {
      m <- with_ctx(paste(lab, "|", clf, "|", mt), {
        tr2 <- make_variant(mt, train, extra_maj, extra_min)
        pr  <- fit_predict(clf, tr2, test)
        perf_metrics(pr, test[[CLASS_COL]])
      })
      perf[[paste(clf, mt)]] <- data.frame(id = job$id, family = job$family,
        IR = job$ir, rep = job$rep, classifier = clf, method = mt, t(m))
      if (mt == "NORES") {                                   # siniflayici kalitesi: yalnizca mudahalesiz egitimde
        ct <- test_calib(pr, test[[CLASS_COL]], tr2)
        cal[[clf]] <- data.frame(id = job$id, classifier = clf,
          oob_bece = co[["bece"]], oob_berr = co[["berr"]], oob_ratio = co[["ratio"]],
          test_bece = ct[["bece"]], test_berr = ct[["berr"]], test_ratio = ct[["ratio"]])
      }
    }
  }
  ## Rakip (literatur) complexity olculeri: siniflayicidan bagimsiz, dengesiz train uzerinde bir kez (ctxl yukarida).
  lit  <- vapply(LIT_NAMES, function(nm)
    with_ctx(paste(lab, "| literatur |", nm), as.numeric(LITERATURE_MEASURES[[nm]](Xl, yl, ctx = ctxl))), numeric(1))
  list(perf = do.call(rbind, perf), cx = do.call(rbind, cx), calib = do.call(rbind, cal),
       lit = data.frame(id = job$id, t(lit)))
}

## ----------------------------- ANA DONGU -----------------------------
jobs <- expand.grid(family = FAMILIES, ir = IR_GRID, rep = seq_len(R_REP),
                    stringsAsFactors = FALSE)
jobs$id   <- seq_len(nrow(jobs))
jobs$seed <- SEED + jobs$id
jobs <- split(jobs, seq_len(nrow(jobs)))

if (file.exists(CKPT)) {
  ck <- readRDS(CKPT)
  if (!identical(ck$sig, SIG))
    stop("Checkpoint farkli ayarlarla (veya eski formatta) uretilmis. Temiz baslamak icin sil: ",
         CKPT, call. = FALSE)
  res <- ck$res; done <- vapply(res, function(r) r$perf$id[1], numeric(1))
  jobs <- Filter(function(j) !(j$id %in% done), jobs)
} else res <- list()

CHUNK <- N_CORES * 4
starts <- if (length(jobs)) seq(1, length(jobs), by = CHUNK) else integer(0)
for (s in starts) {
  idx <- s:min(s + CHUNK - 1, length(jobs))
  ## N_CORES == 1 ise paralel degil: hata tam traceback ile gorunur (once bununla dene)
  out <- if (N_CORES == 1) lapply(jobs[idx], run_one) else
    mclapply(jobs[idx], run_one, mc.cores = N_CORES, mc.preschedule = FALSE)
  bad <- which(!vapply(out, function(o) is.list(o) && !is.null(o$perf), logical(1)))
  if (length(bad)) {
    msg <- out[[bad[1]]]
    stop(sprintf("%d is basarisiz. Ilk hata: %s", length(bad),
                 if (inherits(msg, "try-error")) as.character(msg) else "bilinmeyen (worker coktu mu?)"),
         call. = FALSE)
  }
  res <- c(res, out)
  saveRDS(list(sig = SIG, res = res), CKPT)
  message(sprintf("%d / %d is bitti", length(res), length(res) + length(jobs) - max(idx)))
}

perf <- do.call(rbind, lapply(res, `[[`, "perf"))
cx   <- do.call(rbind, lapply(res, `[[`, "cx"))
calib <- do.call(rbind, lapply(res, `[[`, "calib"))
lit   <- do.call(rbind, lapply(res, `[[`, "lit"))
rownames(perf) <- rownames(cx) <- rownames(calib) <- NULL

## ----------------------------- KAZANC --------------------------------
## Genis tablo: (id, siniflayici) basina bir satir. Tum yontemlerin performanslari
## <METRIK>_<YONTEM> sutunlarinda; kazanc sadece ADD_MAJ / ADD_MIN'den hesaplanir.
wide_perf <- function(mt) {
  d <- perf[perf$method == mt, c("id", "classifier", METRICS)]
  names(d)[-(1:2)] <- paste0(METRICS, "_", mt)
  d
}
gains <- Reduce(function(a, b) merge(a, b, by = c("id", "classifier")),
                lapply(METHODS, wide_perf))
meta  <- unique(perf[, c("id", "family", "IR", "rep", "classifier")])
gains <- merge(meta, gains, by = c("id", "classifier"))
for (m in METRICS) {
  gains[[paste0("gain_maj_", m)]] <- gains[[paste0(m, "_ADD_MAJ")]] - gains[[paste0(m, "_NORES")]]
  gains[[paste0("gain_min_", m)]] <- gains[[paste0(m, "_ADD_MIN")]] - gains[[paste0(m, "_NORES")]]
}
gains <- merge(gains, cx[, c("id", "classifier", CX_NAMES)], by = c("id", "classifier"))
gains <- merge(gains, lit,   by = "id")                       # rakip olculer (veri seti duzeyi; siniflayiciya gore ayni)
gains <- merge(gains, calib, by = c("id", "classifier"))     # oob_* / test_*: bece, berr, ratio (NORES)
stopifnot(!any(duplicated(names(gains))))
gains <- gains[order(gains$id, gains$classifier), ]; rownames(gains) <- NULL

saveRDS(list(perf = perf, cx = cx, calib = calib, lit = lit, gains = gains, settings = list(PRIOR_CORRECTION = PRIOR_CORRECTION)),
        file.path(OUT_DIR, "sim_cxgain_results.rds"))
write.csv(gains, file.path(OUT_DIR, "sim_cxgain_gains.csv"), row.names = FALSE)
write.csv(perf,  file.path(OUT_DIR, "sim_cxgain_perf_long.csv"), row.names = FALSE)

message(sprintf("Bitti: %d satir (id x siniflayici), %d sutun -> %s/",
                nrow(gains), ncol(gains), OUT_DIR))
