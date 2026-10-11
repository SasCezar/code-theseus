library(tidyverse)
library(this.path)
library(patchwork)

script_dir <- dirname(this.path())
base_path  <- file.path(script_dir, "..", "..")

# Written by scripts/python/paper_commit0.py: one row per library, outcome and
# generator ("Pooled" = mean over both generators), with Full - Intent
# (held-out tests already in percentage points).
df <- read_csv(
  file.path(base_path, "data", "out", "benchmark", "agent_libraries_generators.csv"),
  show_col_types = FALSE
) %>%
  filter(outcome %in% c("module_f1", "held_out_pass_rate")) %>%
  mutate(
    outcome = factor(
      recode(outcome,
        "module_f1"          = "Module F1, non-contract",
        "held_out_pass_rate" = "Held-out tests (pp)"
      ),
      levels = c("Module F1, non-contract", "Held-out tests (pp)")
    )
  )

# Libraries sorted by the pooled module F1 gain; those without module F1
# (the contract names every module) at the bottom, by held-out gain.
order <- df %>%
  filter(generator == "Pooled") %>%
  select(name, outcome, diff) %>%
  pivot_wider(names_from = outcome, values_from = diff) %>%
  arrange(!is.na(`Module F1, non-contract`), `Module F1, non-contract`,
          `Held-out tests (pp)`) %>%
  pull(name)
df <- df %>% mutate(name = factor(name, levels = order))

pooled <- df %>%
  filter(generator == "Pooled") %>%
  mutate(direction = factor(
    case_when(diff > 1e-9 ~ "Full higher", diff < -1e-9 ~ "Full lower", TRUE ~ "No change"),
    levels = c("Full higher", "No change", "Full lower")
  ))
single <- df %>% filter(generator != "Pooled")
range_df <- single %>%
  group_by(name, outcome) %>%
  summarise(lo = min(diff), hi = max(diff), .groups = "drop")

FILL_COLORS <- c("Full higher" = "#4E8098", "No change" = "grey60", "Full lower" = "#E07B54")
# Direction is also coded by shape, and the Full-higher side is shaded, so
# the figure reads in black and white
DIR_SHAPES  <- c("Full higher" = 16, "No change" = 4, "Full lower" = 1)
SHAPES      <- c("Gemma" = 2, "Q35" = 0)
# Condition names in small caps, as \cFull and \cIntent in the paper: plotmath
# has no small caps, so the letters after the first are capitals set smaller
sc <- function(word) {
  bquote(.(substr(word, 1, 1)) * scriptstyle(.(toupper(substring(word, 2)))))
}
DELTA_LABEL <- bquote(Delta == .(sc("Full")) - .(sc("Intent")))
# Every legend label and the legend title are plotmath with the same
# vertical extent (vphantom), so they share a baseline: grid centres each
# label on its own bounding box, which differs between plain text and
# plotmath and with the smaller capitals.
on_line     <- function(e) bquote(vphantom("Fg") * .(e))
DIR_LABELS  <- c(
  "Full higher" = on_line(bquote(.(sc("Full")) ~ "higher")),
  "No change"   = on_line("No change"),
  "Full lower"  = on_line(bquote(.(sc("Full")) ~ "lower"))
)
MEAN_TITLE  <- on_line("Mean")
# Generator labels from the manuscript's macros (\mGemma, \mQwen), so the
# legend follows the paper when they change
preamble   <- readLines(file.path(base_path, "paper", "manuscript.tex"))
tex_label  <- function(macro) {
  line <- grep(paste0("\\\\newcommand\\{\\\\", macro, "\\}"), preamble, value = TRUE)
  # \newcommand{\mGemma}{GEMA\xspace} -> GEMA
  sub("^\\\\newcommand\\{[^}]*\\}\\{([^}\\\\]*).*", "\\1", line[1])
}
GEN_LABELS <- c("Gemma" = tex_label("mGemma"), "Q35" = tex_label("mQwen"))

# One panel per outcome, side by side with a shared legend; the subfigure
# tag, (a) or (b), goes under the panel's axis title, centred on the panel.
# The axis title is the difference, Delta = Full - Intent, typeset by plotmath
# (a true minus sign); the tag is a caption, so it keeps its own style.
panel <- function(out, tag, first) {
  ggplot() +
    annotate("rect", xmin = 0, xmax = Inf, ymin = -Inf, ymax = Inf, fill = "grey93") +
    geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50") +
    geom_segment(
      data = filter(range_df, outcome == out),
      aes(x = lo, xend = hi, y = name, yend = name),
      colour = "grey75", linewidth = 0.4
    ) +
    geom_point(
      data = filter(single, outcome == out), aes(x = diff, y = name, shape = generator),
      size = 1.1, colour = "grey35", stroke = 0.4
    ) +
    geom_point(
      data = filter(pooled, outcome == out),
      aes(x = diff, y = name, colour = direction, shape = direction),
      size = 1.8, stroke = 0.8,
      # every key in both panels, so their legends are identical and merge
      show.legend = TRUE
    ) +
    facet_wrap(~outcome) +
    scale_y_discrete(drop = FALSE) +
    scale_colour_manual(
      values = FILL_COLORS, limits = names(FILL_COLORS), name = MEAN_TITLE,
      labels = as.expression(DIR_LABELS)
    ) +
    scale_shape_manual(
      values = c(DIR_SHAPES, SHAPES), limits = c(names(DIR_SHAPES), names(SHAPES)),
      breaks = names(SHAPES), labels = GEN_LABELS[names(SHAPES)], name = NULL
    ) +
    guides(
      # the plotmath labels and title centre lower than plain text, so they
      # are pushed down to sit on their keys like the generator labels
      colour = guide_legend(
        override.aes = list(shape = DIR_SHAPES),
        theme = theme(
          legend.text  = element_text(margin = margin(t = 1.5, l = 2, r = 5)),
          legend.title = element_text(margin = margin(t = 1.5, r = 2))
        )
      ),
      shape  = guide_legend(override.aes = list(size = 1.1, stroke = 0.4, colour = "grey35"))
    ) +
    labs(x = DELTA_LABEL, y = NULL, caption = tag) +
    theme_minimal(base_size = 8) +
    theme(
      axis.title.x       = element_text(margin = margin(t = 1)),
      plot.caption       = element_text(size = rel(1), hjust = 0.5, margin = margin(t = 1)),
      strip.text.x       = element_text(face = "bold", margin = margin(b = 2)),
      plot.margin        = margin(0, 2, 0, 0),
      panel.grid.minor   = element_blank(),
      panel.grid.major.y = element_line(colour = "grey92", linewidth = 0.3),
      axis.text.y        = if (first) element_text(size = 6.5) else element_blank()
    )
}

outcomes <- levels(df$outcome)
p_libraries <- panel(outcomes[1], "(a)", TRUE) + panel(outcomes[2], "(b)", FALSE) +
  plot_layout(guides = "collect") &
  theme(
    legend.position   = "top",
    legend.box        = "vertical",
    legend.spacing.y  = unit(2, "pt"),
    legend.margin     = margin(0, 0, 0, 0),
    legend.text       = element_text(size = 6.5, margin = margin(l = 2, r = 5)),
    legend.title      = element_text(size = 6.5, margin = margin(r = 2)),
    legend.key.width  = unit(6, "pt"),
    legend.key.height = unit(8, "pt"),
    legend.box.margin = margin(0, 0, 0, 0),
    legend.box.spacing = unit(3, "pt"),
    plot.margin       = margin(3, 2, 1, 0)
  )

# Rscript has no screen: print() would write a stray Rplots.pdf
if (interactive()) print(p_libraries)

plots_dir <- file.path(base_path, "paper", "figs")
ggsave(file.path(plots_dir, "agent_libraries.pdf"), p_libraries, width = 3.33, height = 4.4)
ggsave(file.path(plots_dir, "agent_libraries.png"), p_libraries, width = 3.33, height = 4.4, dpi = 300)
