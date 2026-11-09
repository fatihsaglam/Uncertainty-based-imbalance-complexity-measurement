## Banana: bootstrap EU haritasi ile alt-ornek EU haritasi yan yana (her siniflayici icin)
source("global.R"); library(ggplot2); library(gridExtra)

CLFS <- c("RF", "CART", "KNN", "NAIVEBAYES"); M <- 30L; RES <- 150L
NOISE <- 0.2                                      # duzlestirilmis bootstrap: ozellik std. sapmasinin kaci kadar Gauss gurultusu (RF'de yok)
SAMPLE_FRAC <- 0.3                                # < 1: yerine koymadan alt-ornek (geri kalani OOB); bootstrap karsilastirmasi icin sol panel

d   <- as.data.frame(datasets_binary$banana)
cls <- names(d)[sapply(d, function(v) length(unique(v)) == 2)]      # sinif sutunu: 2 farkli degeri olan sutun
stopifnot(length(cls) == 1)
y   <- as.integer(d[[cls]] == names(which.min(table(d[[cls]]))))    # 1 = azinlik
X   <- d[setdiff(names(d), cls)]; xn <- names(X)[1]; yn <- names(X)[2]

rg   <- function(v) seq(min(v) - 0.05 * diff(range(v)), max(v) + 0.05 * diff(range(v)), length.out = RES)
grid <- setNames(expand.grid(rg(X[[1]]), rg(X[[2]])), names(X))

pts <- data.frame(X, sinif = factor(ifelse(y == 1, "azinlik", "cogunluk")))
harita <- function(z, title, brk) {
  ggplot(cbind(grid, z = z), aes(.data[[xn]], .data[[yn]])) +
    geom_contour_filled(aes(z = z), breaks = brk) +
    geom_point(data = pts, aes(colour = sinif, shape = sinif), size = 0.9, alpha = 0.8) +
    scale_fill_viridis_d(name = "EU (bit)", drop = FALSE) +
    scale_colour_manual(values = c(cogunluk = "white", azinlik = "orangered")) +
    coord_cartesian(expand = FALSE) + labs(title = title) + theme_minimal(base_size = 12)
}

for (clf in CLFS) {
  eu_boot  <- complexityCls_uncertainty(X, y, clf, M = M, newdata = grid)$new$eu                                              # bootstrap (replace = TRUE)
  eu_sub   <- complexityCls_uncertainty(X, y, clf, M = M, newdata = grid, sample_frac = SAMPLE_FRAC)$new$eu                    # replace = FALSE
  eu_noise <- if (clf != "RF") complexityCls_uncertainty(X, y, clf, M = M, newdata = grid, noise = NOISE)$new$eu else NULL      # + gurultu
  brk <- pretty(range(eu_boot, eu_sub, eu_noise), 12)                                                                          # tum panellerde ayni renk olcegi
  pl <- list(harita(eu_boot, paste0(clf, ": bootstrap"), brk),
             harita(eu_sub,  paste0(clf, ": alt-ornek %", 100 * SAMPLE_FRAC), brk))
  if (!is.null(eu_noise)) pl[[3]] <- harita(eu_noise, paste0(clf, ": bootstrap + gurultu ", NOISE, " sd"), brk)
  grid.arrange(grobs = pl, ncol = length(pl))
}
