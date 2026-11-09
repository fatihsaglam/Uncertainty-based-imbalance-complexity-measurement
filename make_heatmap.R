## =====================================================================
## make_heatmap.R
## Kesifsel karar izgarasi: EU (sb_min ve sb_diff) x AU (sb_bal), 3x3.
## Her hucre: ortalama SMOTE G-mean kazanci, kazanc > 0.02 olan cift payi, n.
## Renk: pay < %25 "do not oversample", > %75 "oversample", arasi "uncertain".
## Calistirma (proje kok klasorunden): source("make_heatmap.R")
## Girdi : results/real_gains.csv   Cikti: manuscript/figures/fig_real_heatmap.pdf (+ CSV'ler)
## Gerekli paket: ggplot2
## =====================================================================
library(ggplot2)
REAL  <- "results/real_gains.csv"
OUT   <- "manuscript/figures"
DELTA <- 0.02                                   # "degerli" sayilan en kucuk kazanc
LOW <- 0.25; HIGH <- 0.75                       # renk esikleri (hucredeki "degerli" cift payi)
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

r <- read.csv(REAL, stringsAsFactors = FALSE)
r$gain <- r$gain_SMOTE_GMEAN
au_cut <- quantile(r$AUsb_bal, c(0, 1/3, 2/3, 1))
au_cut[1] <- au_cut[1] - 1e-9
r$AU <- cut(r$AUsb_bal, au_cut, labels = c("low", "medium", "high"))

## Panel tanimlari: olcut, esikler, etiket (esikler: Youden esigi ve ikinci kesim)
SPEC <- list(
  list(m = "EUsb_min",  cut = c(0.17, 0.27), tag = "(a)", lab = 'paste("Epistemic measure ", EU[min]^sb)'),
  list(m = "EUsb_diff", cut = c(0.06, 0.20), tag = "(b)", lab = 'paste("Epistemic difference measure ", EU[Delta]^sb)'))

cells <- do.call(rbind, lapply(SPEC, function(s) {
  c1 <- s$cut[1]; c2 <- s$cut[2]
  labs <- c(sprintf("<= %.2f", c1), sprintf("%.2f to %.2f", c1, c2), sprintf("> %.2f", c2))
  e <- cut(r[[s$m]], c(-Inf, c1, c2, Inf), labels = labs)
  d <- data.frame(panel = sprintf('paste("%s ", %s)', s$tag, s$lab), e = e, a = r$AU, g = r$gain)
  out <- expand.grid(e = levels(e), a = levels(r$AU), stringsAsFactors = FALSE)
  out$panel <- d$panel[1]
  out$n     <- mapply(function(ee, aa) sum(d$e == ee & d$a == aa), out$e, out$a)
  out$gain  <- mapply(function(ee, aa) mean(d$g[d$e == ee & d$a == aa]), out$e, out$a)
  out$share <- mapply(function(ee, aa) mean(d$g[d$e == ee & d$a == aa] > DELTA), out$e, out$a)
  out$xpos  <- match(out$e, levels(e))
  out
}))
cells$class <- ifelse(is.na(cells$share), "uncertain",
                ifelse(cells$share < LOW, "do not oversample", ifelse(cells$share > HIGH, "oversample", "uncertain")))
cells$a <- factor(cells$a, levels = c("low", "medium", "high"))   # ggplot: ilk duzey altta
cells$e <- factor(cells$e, levels = unique(cells$e))                # esik sirasini koru
cells$txt <- sprintf("gain %.2f, %.0f%% worth\n(n = %d)", cells$gain, 100 * cells$share, cells$n)
cells$lab <- ifelse(cells$class == "do not oversample", "do not\noversample", cells$class)
cells$ytxt <- as.numeric(cells$a)

cells$measure <- ifelse(grepl("difference", cells$panel), "EUsb_diff", "EUsb_min")
write.csv(cells[, c("measure", "e", "a", "n", "gain", "share", "class")], file.path(OUT, "heatmap_cells.csv"), row.names = FALSE)

p <- ggplot(cells, aes(x = e, y = a, fill = class)) +
  geom_tile(colour = "black", linewidth = 0.4) +
  geom_text(aes(label = lab), nudge_y = 0.12, fontface = "bold", size = 3.6, family = "serif", lineheight = 0.9) +
  geom_text(aes(label = txt), nudge_y = -0.25, size = 2.5, family = "serif", lineheight = 0.9) +
  scale_fill_manual(values = c("do not oversample" = "#F4C7C3", "uncertain" = "#FBE3A6", "oversample" = "#BFE3BF"), guide = "none") +
  facet_wrap(~panel, ncol = 1, scales = "free_x", labeller = label_parsed) +
  labs(x = NULL, y = expression("Aleatoric measure " * AU[bal]^sb)) +
  theme_minimal(base_family = "serif", base_size = 11) +
  theme(panel.grid = element_blank(), strip.text = element_text(hjust = 0.5, size = 10), axis.text = element_text(colour = "black"))
ggsave(file.path(OUT, "fig_real_heatmap.pdf"), p, width = 6.2, height = 7.4)
print(cells[, c("measure", "e", "a", "n", "gain", "share", "class")], digits = 2)
