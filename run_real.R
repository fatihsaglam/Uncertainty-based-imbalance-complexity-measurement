## =====================================================================
## run_real.R  (gercek veri uygulamasi: complexity vs yeniden ornekleme kazanci)
## Amac: Gercek dengesiz veri setlerinde, her veri seti icin
##   (1) 5 katli capraz dogrulama ile siniflayici performansini (yeniden ornekleme olmadan ve SMOTE / ROS ile)
##   (2) her katin EGITIM kismi uzerinde (yeniden orneklemeden ONCE) complexity olculerini
##   toplamak. Katlar uzerinden ortalama alinir; veri setleri arasinda "yeniden ornekleme kazanci" ile
##   complexity korelasyonu sonra incelenir (analiz bu dosyada YOK, sadece veri toplanir).
##
## Tasarim
##  - Veri: data/datasets_binary.rds (179 veri seti). Hedef = SON sutun (2 seviyeli faktor).
##          Azinlik sinifi etiketten degil SAYIDAN belirlenir (az olan sinif = "min").
##  - Secim: n <= MAX_N, p <= MAX_P, tum ongorucular sayisal, azinlik >= MIN_MINORITY.
##          Uygun setlerden IR araliklarina (IR_BINS) gore tabakali N_DATASETS adet secilir (SEED ile tekrarlanabilir).
##  - Capraz dogrulama: K_FOLD katli, sinif oranlarini koruyan (tabakali) bolme.
##  - Yontemler: NORES (dengesiz egitim), SMOTE, ROS (ovRate = 1; k = 5). Test kati dengesiz birakilir (dogal oran).
##  - Metrikler: ROCAUC, MCC, P4, GMEAN.
##  - Complexity (her siniflayici icin AYRI, egitim kati, onsel duzeltmesi KAPALI):
##      EUh_min, EUh_diff, AUh_bal                (5 komsu, sert)
##      EUs_min, EUs_diff, AUs_bal                (15 komsu, yumusak, duz oy)
##      EUsb_min, EUsb_diff, AUsb_bal             (15 komsu, yumusak, agirlikli oy)
##    + tum rakip (literatur) olculer (siniflayicidan bagimsiz).
##  - Kazanc: gain_<SMOTE|ROS>_<METRIK> = METRIK(yontem) - METRIK(NORES)  (katlar uzerinden ortalamalar)
##
## HATA POLITIKASI: siniflayici egitimi / tahmin / yeniden ornekleme hatasi kosuyu DURDURUR (baglam eklenir).
##   Complexity veya literatur olcusu hesaplanamazsa (gercek verilerde tekrarli noktalar, tekil matris vb. olabilir)
##   o deger NA yazilir ve results/real_log.csv'e kaydedilir.
##
## Calistirma: projenin kok klasorunde  source("run_real.R")
##   Once SMOKE <- TRUE ile dene (3 veri seti, 2 kat), sonra FALSE yap.
## Cikti (results/): real_checkpoint.rds, real_results.rds, real_gains.csv, real_selection.csv, real_log.csv
## =====================================================================

source("global.R")            # reticulate/torch, library()'ler ve tum dosyalar (datasets_binary dahil)

suppressPackageStartupMessages({ library(parallel) })

## ----------------------------- AYARLAR -------------------------------
SMOKE        <- FALSE         # TRUE: hizli deneme (3 veri seti, 2 kat)
N_DATASETS   <- 30
IR_BINS      <- c(1, 2, 5, 10, 20, 50, Inf)   # tabakalama icin IR araliklari: (1,2], (2,5], ... , (50, Inf)
MAX_N        <- 3000          # satir siniri (buyuk setlerde RF toplulugu ve n x n mesafe matrisi yavas / bellek yer)
MAX_P        <- 50
MIN_MINORITY <- 25            # 5 kat + SMOTE (k = 5) icin yeterli azinlik
K_FOLD       <- 5
CLASSIFIERS  <- c("LOGREG","NAIVEBAYES","KNN","CART","RF","SVMRADIAL","GBM")
RESAMPLERS   <- c("SMOTE","ROS")
METHODS      <- c("NORES", RESAMPLERS)
METRICS      <- c("ROCAUC","MCC","P4","GMEAN")
N_CORES      <- max(1, detectCores() - 1)    # once 1 ile calistir (hata gorunur)
SEED         <- 2026
OUT_DIR      <- "results"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
CKPT         <- file.path(OUT_DIR, "real_checkpoint.rds")
PRIOR_CORRECTION <- FALSE     # complexity (EU/AU) hesabinda uye basina onsel duzeltmesi (kapali)
CLASS_COL    <- "Class"
POS          <- "min"         # azinlik = pozitif sinif
NEG          <- "maj"
if (SMOKE) { N_DATASETS <- 3; N_CORES <- 1 }
FOLDS_RUN    <- if (SMOKE) 1:2 else seq_len(K_FOLD)

## Agirlikli oylu yumusak surum: complexity_proposed.R'deki fonksiyonlarin balanced = TRUE cagrisi
PROPOSED_MEASURES$EUsb_min  <- function(X, y, ctx = NULL, ...) complexityCls_EUs_min(X, y, ctx = ctx, balanced = TRUE, ...)
PROPOSED_MEASURES$EUsb_diff <- function(X, y, ctx = NULL, ...) complexityCls_EUs_diff(X, y, ctx = ctx, balanced = TRUE, ...)
PROPOSED_MEASURES$AUsb_bal  <- function(X, y, ctx = NULL, ...) complexityCls_AUs_bal(X, y, ctx = ctx, balanced = TRUE, ...)
CX_NAMES  <- c("EUh_min","EUh_diff","AUh_bal", "EUs_min","EUs_diff","AUs_bal", "EUsb_min","EUsb_diff","AUsb_bal")
LIT_NAMES <- setdiff(names(LITERATURE_MEASURES), "IR")

## ----------------------------- VERI HAZIRLAMA ------------------------
## Veri seti -> data.frame(x1..xp, Class); Class: faktor, levels = c("maj","min"); azinlik sayidan belirlenir.
prep_dataset <- function(df) {
  yc <- df[[ncol(df)]]; X <- df[, -ncol(df), drop = FALSE]
  if (!all(vapply(X, is.numeric, logical(1)))) return(NULL)             # sayisal olmayan ongorucu: disarida
  X <- X[, vapply(X, function(v) length(unique(v)) > 1, logical(1)), drop = FALSE]   # sabit sutunlari at
  if (ncol(X) == 0L) return(NULL)
  tb <- table(yc); mn <- names(tb)[which.min(tb)]
  out <- as.data.frame(X); names(out) <- paste0("x", seq_len(ncol(out)))
  out[[CLASS_COL]] <- factor(ifelse(as.character(yc) == mn, POS, NEG), levels = c(NEG, POS))
  out
}
ALL <- lapply(datasets_binary, prep_dataset)
ALL <- ALL[!vapply(ALL, is.null, logical(1))]
INFO <- data.frame(dataset = names(ALL),
                   n = vapply(ALL, nrow, numeric(1)), p = vapply(ALL, function(d) ncol(d) - 1L, numeric(1)),
                   n_min = vapply(ALL, function(d) sum(d[[CLASS_COL]] == POS), numeric(1)),
                   n_maj = vapply(ALL, function(d) sum(d[[CLASS_COL]] == NEG), numeric(1)), stringsAsFactors = FALSE)
INFO$IR <- INFO$n_maj / INFO$n_min

## Tabakali secim: uygun setlerden IR araliklarina gore (esit dagilim, yetmezse kalan araliklardan tamamla)
select_datasets <- function(info, n_sel, seed) {
  el <- info[info$n <= MAX_N & info$p <= MAX_P & info$n_min >= MIN_MINORITY, ]
  set.seed(seed)
  el$bin <- cut(el$IR, IR_BINS, include.lowest = TRUE)
  bins <- levels(el$bin); per <- ceiling(n_sel / length(bins))
  pick <- unlist(lapply(bins, function(b) { ix <- which(el$bin == b); if (length(ix) > per) sample(ix, per) else ix }))
  if (length(pick) > n_sel) pick <- sample(pick, n_sel)
  if (length(pick) < n_sel) { rest <- setdiff(seq_len(nrow(el)), pick); pick <- c(pick, rest[sample.int(length(rest), min(length(rest), n_sel - length(pick)))]) }
  el[sort(pick), ]
}
SEL <- select_datasets(INFO, N_DATASETS, SEED)
write.csv(SEL, file.path(OUT_DIR, "real_selection.csv"), row.names = FALSE)
message(sprintf("Secilen veri seti: %d (uygun: %d / %d); IR araligi %.1f - %.1f", nrow(SEL),
                sum(INFO$n <= MAX_N & INFO$p <= MAX_P & INFO$n_min >= MIN_MINORITY), nrow(INFO), min(SEL$IR), max(SEL$IR)))

## Tabakali K-kat: her sinif kendi icinde karistirilip katlara dagitilir
make_folds <- function(y, k) {
  f <- integer(length(y))
  for (cl in levels(y)) { ix <- which(y == cl); f[ix] <- sample(rep_len(seq_len(k), length(ix))) }
  f
}

## ----------------------------- ADAPTORLER ----------------------------
## (run.R ile ayni: resampler, siniflayici, metrik cagrilari)
RS_ARGS <- list(SMOTE = list(k = 5, ovRate = 1), ROS = list(ovRate = 1))
do_resample <- function(name, train) {
  if (length(unique(table(train[[CLASS_COL]]))) == 1L) return(train)   # zaten dengeli
  x <- train[, setdiff(names(train), CLASS_COL), drop = FALSE]
  r <- do.call(get_resampler(name)$fun, c(list(x = x, y = train[[CLASS_COL]]), RS_ARGS[[name]]))
  out <- as.data.frame(r$x_new); names(out) <- names(x)
  out[[CLASS_COL]] <- factor(r$y_new, levels = c(NEG, POS))
  out
}
make_variant <- function(method, train) {
  out <- switch(method, NORES = train, SMOTE = , ROS = do_resample(method, train), stop("bilinmeyen yontem: ", method))
  rownames(out) <- NULL; out
}
fit_predict <- function(clf_name, train, test) {
  xc  <- setdiff(names(train), CLASS_COL)
  clf <- get_classifier(clf_name)
  clf$fit(train[, xc, drop = FALSE], train[[CLASS_COL]])
  pr <- clf$predict(test[, xc, drop = FALSE])
  list(pred = factor(pr$pred, levels = c(NEG, POS)),
       prob = as.matrix(pr$prob)[, c(NEG, POS), drop = FALSE])
}
perf_metrics <- function(pr, truth) {
  fs <- get_metrics(METRICS)
  vapply(fs, function(f) as.numeric(f(truth = truth, pred = pr$pred, prob = pr$prob,
                                      class_pos = POS, class_neg = NEG)), numeric(1))
}
with_ctx <- function(label, expr) {
  tryCatch(expr, error = function(e) stop(sprintf("[%s] %s", label, conditionMessage(e)), call. = FALSE))
}

## Onerilen complexity (egitim kati, dengesiz): hesaplanamazsa NA + log
get_cx <- function(clf_name, X, y, ctx0, lab) {
  ctx <- ctx0
  unc <- tryCatch(complexityCls_uncertainty(X, y, clf = clf_name, prior_correction = PRIOR_CORRECTION),
                  error = function(e) e)
  if (inherits(unc, "error"))
    return(list(val = setNames(rep(NA_real_, length(CX_NAMES)), CX_NAMES),
                log = data.frame(is = lab, olcut = paste0("belirsizlik|", clf_name), mesaj = conditionMessage(unc))))
  ctx$unc <- unc
  logs <- list()
  val <- vapply(CX_NAMES, function(nm) tryCatch(as.numeric(PROPOSED_MEASURES[[nm]](X, y, ctx = ctx)),
    error = function(e) { logs[[length(logs) + 1L]] <<- data.frame(is = lab, olcut = paste0(nm, "|", clf_name), mesaj = conditionMessage(e)); NA_real_ }),
    numeric(1))
  list(val = val, log = if (length(logs)) do.call(rbind, logs) else NULL)
}

## ----------------------------- TEK IS (veri seti x kat) --------------
run_one <- function(job) {
  d   <- ALL[[job$dataset]]
  set.seed(job$fold_seed); fold <- make_folds(d[[CLASS_COL]], K_FOLD)       # ayni veri seti icin tum katlarda ayni bolme
  set.seed(job$seed)
  lab <- sprintf("id=%d dataset=%s fold=%d", job$id, job$dataset, job$fold)
  train <- d[fold != job$fold, , drop = FALSE]; test <- d[fold == job$fold, , drop = FALSE]
  rownames(train) <- NULL; rownames(test) <- NULL
  Xl <- train[, setdiff(names(train), CLASS_COL), drop = FALSE]; yl <- as.integer(train[[CLASS_COL]] == POS)
  ctxl <- with_ctx(paste(lab, "| complexity baglami"), complexityCls_context(Xl, yl))
  perf <- list(); cx <- list(); logs <- list()
  for (clf in CLASSIFIERS) {
    r <- get_cx(clf, Xl, yl, ctxl, lab)
    cx[[clf]] <- data.frame(id = job$id, dataset = job$dataset, fold = job$fold, classifier = clf, t(r$val))
    if (!is.null(r$log)) logs[[length(logs) + 1L]] <- r$log
    for (mt in METHODS) {
      m <- with_ctx(paste(lab, "|", clf, "|", mt), {
        tr2 <- make_variant(mt, train)
        pr  <- fit_predict(clf, tr2, test)
        perf_metrics(pr, test[[CLASS_COL]])
      })
      perf[[paste(clf, mt)]] <- data.frame(id = job$id, dataset = job$dataset, fold = job$fold,
                                           classifier = clf, method = mt, t(m))
    }
  }
  ## Rakip (literatur) olculer: siniflayicidan bagimsiz, egitim kati uzerinde bir kez
  lit <- vapply(LIT_NAMES, function(nm) tryCatch(as.numeric(LITERATURE_MEASURES[[nm]](Xl, yl, ctx = ctxl)),
    error = function(e) { logs[[length(logs) + 1L]] <<- data.frame(is = lab, olcut = nm, mesaj = conditionMessage(e)); NA_real_ }),
    numeric(1))
  list(perf = do.call(rbind, perf), cx = do.call(rbind, cx),
       lit = data.frame(id = job$id, dataset = job$dataset, fold = job$fold, t(lit)),
       log = if (length(logs)) do.call(rbind, logs) else NULL)
}

## ----------------------------- ANA DONGU -----------------------------
SIG <- list(SEL$dataset, K_FOLD, FOLDS_RUN, CLASSIFIERS, METHODS, METRICS, MAX_N, MAX_P, MIN_MINORITY, SEED,
            LIT_NAMES, CX_NAMES, PRIOR_CORRECTION)
jobs <- expand.grid(dataset = SEL$dataset, fold = FOLDS_RUN, stringsAsFactors = FALSE)
jobs$id <- seq_len(nrow(jobs))
jobs$seed <- SEED + jobs$id
jobs$fold_seed <- SEED + 100000L + match(jobs$dataset, SEL$dataset)
jobs <- split(jobs, seq_len(nrow(jobs)))

if (file.exists(CKPT)) {
  ck <- readRDS(CKPT)
  if (!identical(ck$sig, SIG))
    stop("Checkpoint farkli ayarlarla uretilmis. Temiz baslamak icin sil: ", CKPT, call. = FALSE)
  res <- ck$res; done <- vapply(res, function(r) r$perf$id[1], numeric(1))
  jobs <- Filter(function(j) !(j$id %in% done), jobs)
} else res <- list()

CHUNK <- N_CORES * 4
starts <- if (length(jobs)) seq(1, length(jobs), by = CHUNK) else integer(0)
for (s in starts) {
  idx <- s:min(s + CHUNK - 1, length(jobs))
  out <- if (N_CORES == 1) lapply(jobs[idx], run_one) else
    mclapply(jobs[idx], run_one, mc.cores = N_CORES, mc.preschedule = FALSE)
  bad <- which(!vapply(out, function(o) is.list(o) && !is.null(o$perf), logical(1)))
  if (length(bad)) {
    msg <- out[[bad[1]]]
    stop(sprintf("%d is basarisiz. Ilk hata: %s", length(bad),
                 if (inherits(msg, "try-error")) as.character(msg) else "bilinmeyen (worker coktu mu?)"), call. = FALSE)
  }
  res <- c(res, out)
  saveRDS(list(sig = SIG, res = res), CKPT)
  message(sprintf("%d / %d is bitti", length(res), length(res) + length(jobs) - max(idx)))
}

perf <- do.call(rbind, lapply(res, `[[`, "perf"))
cx   <- do.call(rbind, lapply(res, `[[`, "cx"))
lit  <- do.call(rbind, lapply(res, `[[`, "lit"))
msglog <- do.call(rbind, lapply(res, `[[`, "log"))
rownames(perf) <- rownames(cx) <- rownames(lit) <- NULL

## ----------------------------- KATLAR UZERINDEN ORTALAMA --------------
## veri seti x siniflayici: performans (yontem basina), kazanc, complexity ortalamalari
agg <- function(d, by, cols) aggregate(d[cols], d[by], function(v) mean(v, na.rm = TRUE))
pm <- agg(perf, c("dataset", "classifier", "method"), METRICS)
wide <- Reduce(function(a, b) merge(a, b, by = c("dataset", "classifier")), lapply(METHODS, function(mt) {
  x <- pm[pm$method == mt, c("dataset", "classifier", METRICS)]; names(x)[-(1:2)] <- paste0(METRICS, "_", mt); x }))
for (rs in RESAMPLERS) for (m in METRICS)
  wide[[paste0("gain_", rs, "_", m)]] <- wide[[paste0(m, "_", rs)]] - wide[[paste0(m, "_NORES")]]
cxm  <- agg(cx, c("dataset", "classifier"), CX_NAMES)
litm <- agg(lit, "dataset", LIT_NAMES)
meta <- SEL[, c("dataset", "n", "p", "n_min", "n_maj", "IR")]
gains <- merge(meta, wide, by = "dataset")
gains <- merge(gains, cxm, by = c("dataset", "classifier"))
gains <- merge(gains, litm, by = "dataset")
stopifnot(!any(duplicated(names(gains))))
gains <- gains[order(gains$IR, gains$dataset, gains$classifier), ]; rownames(gains) <- NULL

saveRDS(list(selection = SEL, perf = perf, cx = cx, lit = lit, gains = gains, log = msglog,
             settings = list(K_FOLD = K_FOLD, MAX_N = MAX_N, MAX_P = MAX_P, MIN_MINORITY = MIN_MINORITY, PRIOR_CORRECTION = PRIOR_CORRECTION)),
        file.path(OUT_DIR, "real_results.rds"))
write.csv(gains, file.path(OUT_DIR, "real_gains.csv"), row.names = FALSE)
if (!is.null(msglog)) write.csv(msglog, file.path(OUT_DIR, "real_log.csv"), row.names = FALSE)
message(sprintf("Bitti: %d veri seti x siniflayici satiri, %d sutun -> %s/ (NA olcut kaydi: %d)",
                nrow(gains), ncol(gains), OUT_DIR, if (is.null(msglog)) 0L else nrow(msglog)))
