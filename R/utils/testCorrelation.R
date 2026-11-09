## =====================================================================
## testCorrelation.R
## EU_dir == "min" / "maj" olan satirlarda (none haric) complexity olcutlerinin
## kazanclarla iliskisi. Her complexity icin UC korelasyon:
##   rho_min : complexity ~ gain_min  (azinliga eklemenin kazanci)
##   rho_maj : complexity ~ gain_maj  (cogunluga eklemenin kazanci)
##   rho_dir : complexity ~ yontemin gosterdigi yondeki kazanc
##             (EU_dir == "min" satirda gain_min, "maj" satirda gain_maj)
## =====================================================================

#' @param metric      ZORUNLU ("ROCAUC","MCC","P4","GMEAN" ...).
#' @param complexity  Olcut adi/adlari (ornek "EU_diff"). NULL = gains icindeki tum
#'                    olcutler (EU_min, EU_maj, EU_diff, EU_ratio, AU_bal ... varsa).
#' @param family,classifier,IR  NULL = hepsi (havuzlanir).
#' @param dirs        Hangi EU_dir satirlari: varsayilan c("min","maj"). "none" desteklenmez
#'                    (yon yok, rho_dir tanimsiz).
#' @param method      "spearman" (varsayilan), "kendall" ya da "pearson".
#' @param within      NULL (varsayilan) ya da hucre degiskenleri, ornek c("family","IR"):
#'                    verilirse hem complexity hem kazanc hucre ortalamasindan arindirilir
#'                    (rank uzerinden) -> IR/aile karistiricisi temizlenir. Hucre kaldirildigi
#'                    icin n, hucre sayisi kadar azalmis gibi etki eder; p'ler yaklasiktir.
#' @param by          NULL: havuz. Katman icin ornek by = "EU_dir" ya da c("classifier","EU_dir").
#' @param alpha       Yalnizca `sig_*` isaretleri icin (varsayilan 0.05, duzeltmesiz).
#' @param gains,min_n,verbose  min_n: korelasyon icin gereken en az satir (varsayilan 10).
testCorrelation <- function(metric, complexity = NULL,
                            family = NULL, classifier = NULL, IR = NULL,
                            dirs = c("min", "maj"),
                            method = c("spearman", "kendall", "pearson"),
                            within = NULL, by = NULL, alpha = 0.05,
                            gains = NULL, min_n = 10L, verbose = TRUE) {
  if (missing(metric)) stop("`metric` zorunlu (ornek: metric = \"ROCAUC\").", call. = FALSE)
  dirs   <- match.arg(dirs, c("min", "maj"), several.ok = TRUE)
  method <- match.arg(method)

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

  known_cx <- intersect(c("EU_min","EU_maj","EU_diff","EU_ratio","AU_bal","EU_max","EU_asym","AU_min","AU_maj"),
                        names(gains))
  if (is.null(complexity)) complexity <- known_cx
  bad_cx <- setdiff(complexity, names(gains))
  if (length(bad_cx)) stop("Complexity sutunu yok: ", paste(bad_cx, collapse = ", "),
                           ". Mevcut: ", paste(known_cx, collapse = ", "), call. = FALSE)
  if (!length(complexity)) stop("gains icinde taninan complexity sutunu yok.", call. = FALSE)

  by <- if (is.null(by)) character(0) else by
  bad <- setdiff(c(by, within), names(gains))
  if (length(bad)) stop("by/within icinde bilinmeyen degisken: ", paste(bad, collapse = ", "), call. = FALSE)

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
    warning("Secime uyan EU_dir in {", paste(dirs, collapse = ","),
            "} satiri yok (hepsi \"none\" olabilir). Test yapilmadi, NULL donduruldu.", call. = FALSE)
    return(invisible(NULL))
  }

  gmin <- dat[[paste0("gain_min_", metric)]]; gmaj <- dat[[paste0("gain_maj_", metric)]]
  dat$.g_min <- gmin; dat$.g_maj <- gmaj
  dat$.g_dir <- ifelse(dat$EU_dir == "min", gmin, gmaj)

  ## tek korelasyon: n, rho, p, not
  cor_one <- function(x, y, cell) {
    out <- list(n = NA_integer_, rho = NA_real_, p = NA_real_, note = "")
    ok <- is.finite(x) & is.finite(y); x <- x[ok]; y <- y[ok]; cell <- cell[ok]
    out$n <- length(x)
    if (length(x) < min_n) { out$note <- sprintf("n < %d", min_n); return(out) }
    if (!is.null(within)) {                      # hucre ici arindirma (rank uzerinden)
      if (method != "pearson") { x <- rank(x); y <- rank(y) }
      x <- x - ave(x, cell); y <- y - ave(y, cell)
    }
    if (sd(x) == 0 || sd(y) == 0) { out$note <- "sabit degisken (sd = 0)"; return(out) }
    m <- if (!is.null(within)) "pearson" else method
    ct <- cor.test(x, y, method = m, exact = FALSE)
    out$rho <- unname(ct$estimate); out$p <- ct$p.value
    out
  }

  one_stratum <- function(x) {
    cell <- if (is.null(within)) rep(1L, nrow(x)) else interaction(x[within], drop = TRUE)
    do.call(rbind, lapply(complexity, function(cx) {
      a <- cor_one(x[[cx]], x$.g_min, cell)
      b <- cor_one(x[[cx]], x$.g_maj, cell)
      d <- cor_one(x[[cx]], x$.g_dir, cell)
      notes <- unique(c(a$note, b$note, d$note)); notes <- notes[nzchar(notes)]
      data.frame(complexity = cx, n = d$n,
                 rho_min = a$rho, p_min = a$p,
                 rho_maj = b$rho, p_maj = b$p,
                 rho_dir = d$rho, p_dir = d$p,
                 note = paste(notes, collapse = "; "), stringsAsFactors = FALSE)
    }))
  }

  grp <- if (length(by)) split(dat, dat[by], drop = TRUE) else list(all = dat)
  res <- do.call(rbind, lapply(grp, function(x) {
    r <- one_stratum(x)
    if (length(by)) cbind(x[1L, by, drop = FALSE], metric = metric, r) else cbind(metric = metric, r)
  }))
  rownames(res) <- NULL
  res$sig_dir <- res$p_dir < alpha

  if (verbose) {
    cat(sprintf("\nKorelasyon (%s%s) | metrik = %s | EU_dir in {%s}\n", method,
                if (!is.null(within)) paste0(", hucre-ici: ", paste(within, collapse = " x ")) else "",
                metric, paste(dirs, collapse = ",")))
    cat(sprintf("Katman: %s | secilen satir: %d\n",
                if (length(by)) paste(by, collapse = " x ") else "havuz", nrow(dat)))
    print(res, digits = 3, row.names = FALSE)
    cat("\nrho_min / rho_maj: complexity ile azinliga / cogunluga eklemenin kazanci\n")
    cat("rho_dir: complexity ile yontemin gosterdigi yondeki kazanc\n")
    cat("Not: p'ler duzeltmesiz; satirlar bagimsiz degil (ayni train, birden fazla siniflayici).\n")
  }
  invisible(res)
}
