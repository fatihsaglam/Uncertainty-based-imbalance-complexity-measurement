## =====================================================================
## testEquivalence.R
## EU_dir == "none" olan satirlarda, veri eklemenin performansi degistirmedigini
## (kazancin 0'a esdeger oldugunu) TOST ile sinar.
##
## Neden TOST: klasik t testinde anlamsiz sonuc "etki yok" demek degildir;
## TOST H0'i "etki >= margin" olarak kurar, reddedilirse |ortalama kazanc| < margin
## (esdeger / ihmal edilebilir) sonucuna varilir.
##
## Veri: kazanc = performans(sonra) - performans(once) zaten sim_complexity_vs_gain.R
## ciktisinda var (gain_min_<metrik>, gain_maj_<metrik>); ayni egitim ve test verisinden
## geldigi icin ESLESTIRILMIS farklardir, test bu farklar uzerinde yapilir.
## =====================================================================

#' @param metric  ZORUNLU. Metrik adi: "ROCAUC", "MCC", "P4", "GMEAN" ...
#' @param family,classifier,IR  Secilecek deger(ler). NULL (varsayilan) = hepsi.
#' @param gain_direction  Hangi kazanc: "min" (azinliga veri eklenince), "maj" (cogunluga).
#'                Varsayilan ikisi birden.
#' @param margin  Esdegerlik toleransi (metrik biriminde, +-margin). VARSAYILAN 0.01
#'                bir secimdir; "ihmal edilebilir kazanc" tanimi sana ait.
#' @param alpha   Her iki tek tarafli testin duzeyi (varsayilan 0.05; %90 GA ile ayni).
#' @param by      Varsayilan NULL: secilen alt kumedeki TUM none satirlari havuzlanir,
#'                kazanc yonu basina TEK test yapilir. Katman katman istersen
#'                by = c("family","classifier","IR") ya da bir kismini ver.
#' @param gains   gains data.frame'i. NULL ise results/sim_cxgain_results.rds okunur.
#' @param min_n   Bir katmanda test icin gereken en az "none" satiri (varsayilan 3).
#' @param method  "t" (varsayilan): t tabanli TOST. "boot": yuzdelik bootstrap TOST — ortalama
#'                kazanc icin (1-2*alpha) bootstrap araligi; ±margin icindeyse esdeger.
#'                p_lower/p_upper = (B-ort. <= -margin / >= +margin sayisi + 1)/(B + 1).
#'                gains icinde `id` sutunu varsa KUME bootstrap (id'ye gore): ayni egitim
#'                verisini paylasan satirlar (farkli siniflayicilar) birlikte cekilir.
#' @param B       Bootstrap sayisi (varsayilan 5000). @param seed Tekrarlanabilirlik (NULL = rastgele).
#' @param verbose TRUE ise tabloyu ekrana basar.
#' @return data.frame (invisible): katman basina bir satir (kazanc turu basina).
testEquivalence <- function(metric,
                            family = NULL, classifier = NULL, IR = NULL,
                            gain_direction = c("min", "maj"),
                            margin = 0.01, alpha = 0.05,
                            by = NULL,
                            gains = NULL, min_n = 3L, verbose = TRUE,
                            method = c("t", "boot"), B = 5000L, seed = 1L) {
  if (missing(metric)) stop("`metric` zorunlu (ornek: metric = \"ROCAUC\").", call. = FALSE)
  gain_direction <- match.arg(gain_direction, c("min", "maj"), several.ok = TRUE)
  method <- match.arg(method)
  if (method == "boot" && !is.null(seed)) {                # global RNG durumunu koru
    had <- exists(".Random.seed", envir = globalenv(), inherits = FALSE)
    if (had) old_seed <- get(".Random.seed", envir = globalenv())
    on.exit(if (had) assign(".Random.seed", old_seed, envir = globalenv())
            else if (exists(".Random.seed", envir = globalenv(), inherits = FALSE))
              rm(".Random.seed", envir = globalenv()), add = TRUE)
    set.seed(seed)
  }
  stopifnot(is.numeric(margin), margin > 0, alpha > 0, alpha < 1)
  
  if (is.null(gains)) {
    dir <- if (exists("OUT_DIR", inherits = TRUE)) get("OUT_DIR") else "results"
    f <- file.path(dir, "sim_cxgain_results.rds")
    if (!file.exists(f)) stop("gains verilmedi ve ", f, " bulunamadi.", call. = FALSE)
    gains <- readRDS(f)$gains
  }
  
  ## ---- dogrulamalar (sessiz gecme yok) ----
  gcols <- paste0("gain_", gain_direction, "_", metric)
  miss <- setdiff(gcols, names(gains))
  if (length(miss)) {
    avail <- sub("^gain_(min|maj)_", "", grep("^gain_min_", names(gains), value = TRUE))
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
  
  ## ---- secim: sadece "none" satirlari + istenen family/classifier/IR ----
  dat <- gains[gains$EU_dir == "none", , drop = FALSE]
  if (!is.null(family))     dat <- dat[dat$family     %in% family, , drop = FALSE]
  if (!is.null(classifier)) dat <- dat[dat$classifier %in% classifier, , drop = FALSE]
  if (!is.null(IR))         dat <- dat[dat$IR         %in% IR, , drop = FALSE]
  if (nrow(dat) == 0L) {
    sel <- c(if (!is.null(family)) paste0("family=", paste(family, collapse = "/")),
             if (!is.null(classifier)) paste0("classifier=", paste(classifier, collapse = "/")),
             if (!is.null(IR)) paste0("IR=", paste(IR, collapse = "/")))
    n_sel <- sum(Reduce(`&`, c(list(rep(TRUE, nrow(gains))),
                               list(if (!is.null(family)) gains$family %in% family else TRUE,
                                    if (!is.null(classifier)) gains$classifier %in% classifier else TRUE,
                                    if (!is.null(IR)) gains$IR %in% IR else TRUE))))
    warning("Secime (", if (length(sel)) paste(sel, collapse = ", ") else "tum veri",
            ") uyan EU_dir == \"none\" satiri yok (secimdeki toplam satir: ", n_sel,
            "; hepsi min/maj olarak etiketlenmis). Test yapilmadi, NULL donduruldu.", call. = FALSE)
    return(invisible(NULL))
  }
  
  ## ---- tek katman icin eslestirilmis TOST ----
  boot_means <- function(d, id) {                          # kume bootstrap ortalamalari
    f <- factor(if (is.null(id)) seq_along(d) else id)
    sm <- as.numeric(tapply(d, f, sum)); ct <- as.numeric(tapply(d, f, length)); K <- length(sm)
    idx <- matrix(sample.int(K, K * B, replace = TRUE), K, B)
    colSums(matrix(sm[idx], K, B)) / colSums(matrix(ct[idx], K, B))
  }
  tost_one <- function(d, id = NULL) {
    ok <- is.finite(d); d <- d[ok]; if (!is.null(id)) id <- id[ok]
    n <- length(d)
    out <- list(n = n, mean_diff = NA_real_, sd_diff = NA_real_, se = NA_real_,
                ci_low = NA_real_, ci_high = NA_real_,
                p_lower = NA_real_, p_upper = NA_real_, p_tost = NA_real_,
                equivalent = NA, p_ttest = NA_real_, note = "")
    if (n < min_n) { out$note <- sprintf("n < %d", min_n); return(out) }
    m <- mean(d); s <- sd(d); se <- s / sqrt(n); df <- n - 1
    out$mean_diff <- m; out$sd_diff <- s; out$se <- se
    if (!is.finite(se) || se == 0) {                       # tum farklar ayni (orn. hepsi 0)
      out$ci_low <- out$ci_high <- m
      out$p_lower <- out$p_upper <- out$p_tost <- if (abs(m) < margin) 0 else 1
      out$equivalent <- abs(m) < margin
      out$note <- "sd = 0"
      return(out)
    }
    tq <- qt(1 - alpha, df)
    out$ci_low  <- m - tq * se                             # (1 - 2*alpha) GA
    out$ci_high <- m + tq * se
    out$p_lower <- pt((m + margin) / se, df, lower.tail = FALSE)   # H0: ort <= -margin
    out$p_upper <- pt((m - margin) / se, df, lower.tail = TRUE)    # H0: ort >= +margin
    out$p_tost  <- max(out$p_lower, out$p_upper)
    out$equivalent <- out$p_tost < alpha
    out$p_ttest <- 2 * pt(-abs(m / se), df)                # klasik H0: ort = 0 (karsilastirma icin)
    if (method == "boot") {
      bm <- boot_means(d, id)
      out$se      <- sd(bm)                                # bootstrap SE
      out$ci_low  <- unname(quantile(bm, alpha));  out$ci_high <- unname(quantile(bm, 1 - alpha))
      out$p_lower <- (sum(bm <= -margin) + 1) / (B + 1)
      out$p_upper <- (sum(bm >=  margin) + 1) / (B + 1)
      out$p_tost  <- max(out$p_lower, out$p_upper)
      out$equivalent <- out$p_tost < alpha
      out$note <- if (!is.null(id) && anyDuplicated(id)) "kume bootstrap (id)" else ""
    }
    out
  }
  
  ## ---- katmanlara bol, her kazanc turu icin calistir ----
  dir_col <- "gain_direction"      # sonuc tablosundaki kazanc yonu sutunu (tek tanim)
  one_gain <- function(g) {
    col <- paste0("gain_", g, "_", metric)
    grp <- if (length(by)) split(dat, dat[by], drop = TRUE) else list(all = dat)
    rows <- lapply(grp, function(x) {
      base <- if (length(by)) x[1L, by, drop = FALSE] else data.frame(row.names = 1L)
      meta <- setNames(data.frame(metric, g, margin, stringsAsFactors = FALSE),
                       c("metric", dir_col, "margin"))
      cbind(base, meta, as.data.frame(tost_one(x[[col]], if ("id" %in% names(x)) x$id else NULL),
                                      stringsAsFactors = FALSE))
    })
    do.call(rbind, rows)
  }
  res <- do.call(rbind, lapply(gain_direction, one_gain))
  rownames(res) <- NULL
  ord <- c(by, dir_col)
  if (!all(ord %in% names(res)))
    stop("Siralama sutunlari sonuc tablosunda yok: ", paste(setdiff(ord, names(res)), collapse = ", "),
         " | tablo sutunlari: ", paste(names(res), collapse = ", "), call. = FALSE)
  res <- res[do.call(order, unname(as.list(res[ord]))), , drop = FALSE]
  rownames(res) <- NULL
  
  if (verbose) {
    cat(sprintf("\nTOST (eslestirilmis, %s) | metrik = %s | margin = +-%g | alpha = %g | EU_dir = none\n",
                if (method == "boot") sprintf("bootstrap B=%d", B) else "t", metric, margin, alpha))
    cat(sprintf("Katman: %s | secilen none satiri: %d\n",
                if (length(by)) paste(by, collapse = " x ") else "havuz", nrow(dat)))
    print(res, digits = 3, row.names = FALSE)
    tested <- !is.na(res$equivalent)
    cat(sprintf("\nTest edilen katman: %d | esdeger bulunan: %d | esdeger bulunamayan: %d | n yetersiz: %d\n",
                sum(tested), sum(res$equivalent %in% TRUE), sum(res$equivalent %in% FALSE),
                sum(!tested)))
    cat("Not: esdeger bulunamadi = 'etki var' degil, 'bu margin icinde oldugu gosterilemedi'.\n")
  }
  invisible(res)
}