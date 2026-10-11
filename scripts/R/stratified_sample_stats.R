library(tidyverse)
library(this.path)
library(ggh4x)

script_dir <- dirname(this.path())
base_path <- file.path(script_dir, "..", "..")

strata <- readr::read_csv(
  file.path(
    base_path, "data", "out",
    "reverse-se_v0.6_p10-95_ctx95_dataset_sample_1000_stratified.strata_stats.csv"
  ),
  show_col_types = FALSE
) %>%
  mutate(
    language = as.factor(language),
    size_bin = factor(size_bin, levels = c("small", "medium", "large")),
    star_bin = factor(star_bin, levels = c("low", "medium", "high"))
  )

# Bin edges mirror SIZE_BIN_EDGES / STAR_BIN_EDGES in
# scripts/python/stratified_sample.py -- keep these in sync if those change.
size_bin_labels <- c(
  small  = "Small\n(< 100)",
  medium = "Medium\n(100–1,000)",
  large  = "Large\n(≥ 1,000)"
)
star_bin_labels <- c(
  low    = "Low\n(< 500)",
  medium = "Medium\n(500–5,000)",
  high   = "High\n(≥ 5,000)"
)

sample_mid <- (min(strata$sample) + max(strata$sample)) / 2

strata_labelled <- strata %>%
  mutate(
    text_colour = if_else(sample > sample_mid, "white", "black"),
    star_header = "# Stars"
  )

p_all <- ggplot(strata_labelled, aes(x = language, y = size_bin, fill = sample)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sample, color = text_colour), size = 5) +
  scale_color_identity() +
  facet_nested(
    . ~ star_header + star_bin,
    labeller = labeller(star_bin = as_labeller(function(x) star_bin_labels[x])),
    strip = strip_nested(
      text_x = elem_list_text(
        size = c(18, 14.4),
        face = "plain",
        margin = list(
          margin(t = 2, b = 2, unit = "pt"),
          margin(t = 2, b = 2, unit = "pt")
        )
      ),
      by_layer_x = TRUE
    )
  ) +
  scale_fill_gradient(low = "#d6e9f7", high = "#08306b", name = "# Repos") +
  scale_y_discrete(labels = size_bin_labels) +
  labs(x = "Language", y = "# Nodes") +
  theme_minimal(base_size = 18) +
  theme(
    panel.grid = element_blank(),
    panel.border = element_blank(),
    panel.background = element_blank(),
    plot.background = element_blank(),
    panel.spacing = unit(1.2, "lines"),
    strip.background = element_blank(),
    legend.margin = margin(0, 0, 0, 0),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )
# Rscript has no screen: printing a plot would write a stray Rplots.pdf
if (interactive()) print(p_all)

ggsave(file.path(base_path, "figures", "plots", "stratified_sample_all_strata.pdf"), p_all, width = 9, height = 4)
ggsave(file.path(base_path, "figures", "plots", "stratified_sample_all_strata.png"), p_all, width = 9, height = 4, dpi = 300)
