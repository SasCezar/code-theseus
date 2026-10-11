library(tidyverse)
library(scales)
library(this.path)

script_dir <- dirname(this.path())
base_path  <- file.path(script_dir, "..", "..")

# Written by scripts/python/paper_commit0.py: one row per library and outcome,
# with the Full - Intent difference and the Spearman correlations.
df <- read_csv(
  file.path(base_path, "data", "out", "benchmark", "agent_contract_libraries.csv"),
  show_col_types = FALSE
) %>%
  filter(outcome %in% c("module_f1", "held_out_pass_rate")) %>%
  mutate(
    outcome = recode(outcome,
      "module_f1"          = "Module F1, non-contract",
      "held_out_pass_rate" = "Held-out tests passed (pp)"
    ),
    outcome = factor(outcome, levels = c("Module F1, non-contract", "Held-out tests passed (pp)")),
    direction = factor(
      case_when(diff > 1e-9 ~ "Full higher", diff < -1e-9 ~ "Full lower", TRUE ~ "No change"),
      levels = c("Full higher", "No change", "Full lower")
    )
  )

# ── helpers ───────────────────────────────────────────────────────────────────

FILL_COLORS <- c("Full higher" = "#4E8098", "No change" = "grey60", "Full lower" = "#E07B54")
# Same shapes as agent_libraries.R, so direction reads in black and white
DIR_SHAPES  <- c("Full higher" = 16, "No change" = 4, "Full lower" = 1)
# Condition names in small caps, as \cFull and \cIntent in the paper: plotmath
# has no small caps, so the letters after the first are capitals set smaller
sc <- function(word) {
  bquote(.(substr(word, 1, 1)) * scriptstyle(.(toupper(substring(word, 2)))))
}
DELTA_LABEL <- bquote(Delta == .(sc("Full")) - .(sc("Intent")))
# Every legend label is plotmath with the same vertical extent (vphantom),
# so they share a baseline: grid centres each label on its own bounding box,
# which differs between plain text and plotmath and with the smaller capitals.
on_line     <- function(e) bquote(vphantom("Fg") * .(e))
DIR_LABELS  <- c(
  "Full higher" = on_line(bquote(.(sc("Full")) ~ "higher")),
  "No change"   = on_line("No change"),
  "Full lower"  = on_line(bquote(.(sc("Full")) ~ "lower"))
)

base_theme <- function() {
  theme_minimal(base_size = 9) +
    theme(
      legend.position  = "top",
      legend.title     = element_blank(),
      # plotmath labels centre lower than plain text: push them onto the keys
      legend.text      = element_text(margin = margin(t = 1.5, l = 5.5, r = 5.5)),
      strip.text       = element_text(face = "bold"),
      panel.grid.minor = element_blank(),
      legend.margin    = margin(0, 0, 0, 0),
      legend.box.margin = margin(0, 0, -6, 0)
    )
}

# plotmath, so that rho renders with the default pdf device
p_label <- function(p) {
  if_else(p < 0.001, "italic(p) < '.001'", paste0("italic(p) == '", sub("^0", "", sprintf("%.3f", p)), "'"))
}

labels <- df %>%
  distinct(outcome, rho, rho_p) %>%
  mutate(label = paste0("rho == '", sprintf("%.2f", rho), "' * ',' ~ ", p_label(rho_p)))

# ── Full - Intent vs contract coverage ────────────────────────────────────────

p_contract <- ggplot(df, aes(x = cov, y = diff)) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50") +
  geom_smooth(method = "lm", formula = y ~ x, colour = "grey30", fill = "grey80", linewidth = 0.6) +
  geom_point(
    aes(colour = direction, shape = direction, size = reference_modules),
    alpha = 0.8, stroke = 0.8
  ) +
  geom_text(
    data = labels, aes(x = -Inf, y = Inf, label = label),
    hjust = -0.05, vjust = 1.4, size = 2.9, parse = TRUE, inherit.aes = FALSE
  ) +
  facet_wrap(~outcome, ncol = 1, scales = "free_y") +
  scale_colour_manual(values = FILL_COLORS, labels = as.expression(DIR_LABELS)) +
  scale_shape_manual(values = DIR_SHAPES, labels = as.expression(DIR_LABELS)) +
  scale_size_area(max_size = 3, guide = "none") +
  scale_x_continuous(labels = percent, limits = c(0, 1)) +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.22))) +
  labs(x = "Contract coverage", y = DELTA_LABEL) +
  base_theme()

# Rscript has no screen: print() would write a stray Rplots.pdf
if (interactive()) print(p_contract)

# ── save ──────────────────────────────────────────────────────────────────────

plots_dir <- file.path(base_path, "paper", "figs")
ggsave(file.path(plots_dir, "agent_contract.pdf"), p_contract, width = 3.33, height = 3.6)
ggsave(file.path(plots_dir, "agent_contract.png"), p_contract, width = 3.33, height = 3.6, dpi = 300)
