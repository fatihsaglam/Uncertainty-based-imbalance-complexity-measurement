## =====================================================================
## testDirection.R
## EU_dir == "min" / "maj" olan satirlarda, YONTEMIN GOSTERDIGI sinifa veri eklemenin
## gercekten kazanc getirip getirmedigini sinar (testEquivalence'in "none" tamamlayicisi).
##   EU_dir = "min" -> kazanc = gain_min_<metrik>   (digeri: gain_maj_<metrik>)
##   EU_dir = "maj" -> kazanc = gain_maj_<metrik>   (digeri: gain_min_<metrik>)
## Her satirda "gosterilen yon" ve "diger yon" kazanci eslestirilmis (ayni train/test).
##
## Testler (hepsi tek tarafli, eslestirilmis t):
##  (a) p_gt0      : H0: ort. kazanc <= 0       -> kazanc pozitif mi?
##  (b) p_gt_margin: H0: ort. kazanc <= margin  -> kazanc ANLAMLI buyuklukte mi? (relevant)
##  (c) p_vs_other : H0: ort.(gosterilen - diger) <= 0 -> yontemin gosterdigi sinifa eklemek,
##                   digerine eklemekten daha iyi mi? (yonun asil dogrulamasi)
## =====================================================================

#' @param metric  ZORUNLU ("ROCAUC","MCC","P4","GMEAN" ...).
#' @param family,classifier,IR  NULL = hepsi (havuzlanir).
#' @param dirs    Hangi EU_dir degerleri: varsayilan c("min","maj"), her biri icin ayri test.
#' @param margin  "Anlamli kazanc" esigi (metrik biriminde). Varsayilan 0.01 senin tanimin.
#' @param alpha   Tek tarafli test duzeyi (varsayilan 0.05).
#' @param by      Varsayilan NULL: havuz, EU_dir basina tek test. Katman icin
#'                by = c("family","classifier","IR") (ya da bir kismi).
#' @param gains,min_n,verbose  testEquivalence ile ayni.
testDirection <- function(metric,
                          family = NULL, classifier = NULL, IR = NULL,
                          dirs = c("min", "maj"),
                          margin = 0.01, alpha = 0.05,
                          by = NULL, gains = NULL, min_n = 3L, verbose = TRUE) {
  if (missing(metric)) stop("`metric` zorunlu (ornek: metric = \"ROCAUC\").", call. = FALSE)
  dirs <- match.arg(dirs, c("min", "maj"), several.ok = TRUE)
  stopifnot(is.numeric(margin), margin > 0, alpha > 0, alpha < 1)

  if (is.null(gains)) {
    dir <- if (exists("OUT_DIR", inherits = TRUE)) get("OUT_DIR") else "results"
    f <- file.path(dir, "sim_cxgain_results.rds")
    if (!file.exists(f)) stop("gains verilmedi ve ", f, " bulunamadi.", call. = FALSE)
    gains <- readRDS(f)$gains
  }
  gcols <- paste0("gain_", c("min", "maj"), "_", metric)
  miss <- setdiff(gcols, names(gains))
  if (length(miss)) {
    avail <- sub("^gain_min_", "", grep("^gain_min_", names(gains), value = TRUE))
    stop("Sutun yok: ", paste(miss, collapse = ", "),
         ". Mevcut metrikler: ", paste(avail, collapse = ", "), call. = FALSE)
  }
  if (!"EU_dir" %in% names(gains)) stop("gains icinde EU_dir sutunu yok.", call. = FALSE)
  by <- if (is.null(by)) character(0) else by
  bad_by <- setdiff(by, names(gains))
  if (length(bad_by)) stop("by icinde bilinmeyen degisken: ", paste(bad_by, collapse = ", "), call. = FALSE)
  check_sel <- function(x, nm) {
    if (is.null(x)) return(invisible())
    unk <- setdiff(as.character(x), as.character(unique(gains[[nm]])))
    if (length(unk)) stop(nm, " icin veride olmayan deger(ler): ", paste(unk, collapse = ", "),
                          ". Mevcut: ", paste(sort(unique(as.character(gains[[nm]]))), collapse = ", "),
                          call. = FALSE)
  }
  check_sel(family, "family"); check_sel(classifier, "classifier"); check_sel(IR, "IR")

  dat <- gains[gains$EU_dir %in% dirs, , drop = FALSE]
  if (!is.null(family))     dat <- dat[dat$family     %in% family, , drop = FALSE]
  if (!is.null(classifier)) dat <- dat[dat$classifier %in% classifier, , drop = FALSE]
  if (!is.null(IR))         dat <- dat[dat$IR         %in% IR, , drop = FALSE]
  if (nrow(dat) == 0L) {
    sel <- c(if (!is.null(family)) paste0("family=", paste(family, collapse = "/")),
             if (!is.null(classifier)) paste0("classifier=", paste(classifier, collapse = "/")),
             if (!is.null(IR)) paste0("IR=", paste(IR, collapse = "/")))
    warning("Secime (", if (length(sel)) paste(sel, collapse = ", ") else "tum veri",
            ") uyan EU_dir in {", paste(dirs, collapse = ","), "} satiri yok (hepsi \"none\" olabilir). ",
            "Test yapilmadi, NULL donduruldu.", call. = FALSE)
    return(invisible(NULL))
  }

  ## gosterilen / diger yon kazanci
  gmin <- dat[[paste0("gain_min_", metric)]]; gmaj <- dat[[paste0("gain_maj_", metric)]]
  dat$.g_ind   <- ifelse(dat$EU_dir == "min", gmin, gmaj)
  dat$.g_other <- ifelse(dat$EU_dir == "min", gmaj, gmin)

  ## tek tarafli t: H0: ort <= mu0  (H1: ort > mu0)
  ## sd = 0 durumunda p, ortalama mu0'in ustundeyse 0, degilse 1
  p_greater <- function(d, mu0) {
    n <- length(d); m <- mean(d); s <- sd(d)
    if (!is.finite(s) || s == 0) return(if (m > mu0) 0 else 1)
    pt((m - mu0) / (s / sqrt(n)), n - 1, lower.tail = FALSE)
  }
  test_one <- function(g_ind, g_other) {
    ok <- is.finite(g_ind) & is.finite(g_other); g_ind <- g_ind[ok]; g_other <- g_other[ok]
    n <- length(g_ind)
    out <- list(n = n, mean_gain = NA_real_, ci_low = NA_real_, mean_other = NA_real_,
                mean_contrast = NA_real_, p_gt0 = NA_real_, p_gt_margin = NA_real_,
                relevant = NA, p_vs_other = NA_real_, better_than_other = NA, note = "")
    if (n < min_n) { out$note <- sprintf("n < %d", min_n); return(out) }
    s <- sd(g_ind); se <- s / sqrt(n)
    out$mean_gain     <- mean(g_ind)
    out$ci_low        <- if (is.finite(se) && se > 0) mean(g_ind) - qt(1 - alpha, n - 1) * se else mean(g_ind)
    out$mean_other    <- mean(g_other)
    out$mean_contrast <- mean(g_ind - g_other)
    out$p_gt0         <- p_greater(g_ind, 0)
    out$p_gt_margin   <- p_greater(g_ind, margin)
    out$relevant      <- out$p_gt_margin < alpha
    out$p_vs_other    <- p_greater(g_ind - g_other, 0)
    out$better_than_other <- out$p_vs_other < alpha
    if (!is.finite(se) || se == 0) out$note <- "sd = 0"
    out
  }

  keys <- c("EU_dir", by)
  grp <- split(dat, dat[keys], drop = TRUE)
  res <- do.call(rbind, lapply(grp, function(x) {
    cbind(x[1L, keys, drop = FALSE], metric = metric, margin = margin,
          as.data.frame(test_one(x$.g_ind, x$.g_other), stringsAsFactors = FALSE))
  }))
  rownames(res) <- NULL
  res <- res[do.call(order, unname(as.list(res[keys]))), , drop = FALSE]
  rownames(res) <- NULL

  if (verbose) {
    cat(sprintf("\nYon testi (eslestirilmis, tek tarafli) | metrik = %s | anlamli kazanc > %g | alpha = %g\n",
                metric, margin, alpha))
    cat("Kazanc = yontemin gosterdigi sinifa (EU_dir) 300 gercek ornek eklemenin getirisi.\n")
    cat(sprintf("Katman: EU_dir%s | secilen satir: %d\n",
                if (length(by)) paste0(" x ", paste(by, collapse = " x ")) else " (havuz)", nrow(dat)))
    print(res, digits = 3, row.names = FALSE)
    cat("\nmean_gain: gosterilen sinifa eklemenin ort. kazanci (ci_low: tek tarafli alt sinir)\n")
    cat("mean_other: ayni satirlarda DIGER sinifa eklemenin ort. kazanci; mean_contrast = fark\n")
    cat("relevant = p_gt_margin < alpha  |  better_than_other = p_vs_other < alpha\n")
  }
  invisible(res)
}
