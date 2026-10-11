library(tidyverse)
library(scales)
library(hexbin)
library(this.path)

script_dir <- dirname(this.path())
base_path  <- file.path(script_dir, "..", "..")

df_stats <- read_csv(
  file.path(base_path, "data", "out", "reverse-se_v0.6_p10-95_ctx95_stats.csv"),
  show_col_types = FALSE
) %>%
  mutate(filtered = as.logical(filtered))

df <- read_csv(
  file.path(base_path, "data", "out", "reverse-se_v0.6_p10-95_ctx95_tokens.csv"),
  show_col_types = FALSE
) %>%
  mutate(
    graph_type = recode(graph_type,
      "original"     = "Raw",
      "package_only" = "Compressed"
    ),
    graph_type = factor(graph_type, levels = c("Raw", "Compressed"))
  ) %>%
  left_join(
    df_stats %>% select(repo, node_compression_ratio, edge_compression_ratio, filtered),
    by = c("repo_id" = "repo")
  ) %>%
  filter(!filtered)

# ── helpers ───────────────────────────────────────────────────────────────────

# Distinct from the Full higher / lower colours of the agent figures
FILL_COLORS <- c("Raw" = "#7d7d7d", "Compressed" = "#c7798f")
# Raw is also dashed, so the two curves read in black and white
LINE_TYPES  <- c("Raw" = "22", "Compressed" = "solid")

base_theme <- function() {
  theme_minimal(base_size = 11) +
    theme(
      legend.position  = "top",
      legend.title     = element_blank(),
      strip.text       = element_text(face = "bold"),
      panel.grid.minor = element_blank()
    )
}

density_plot <- function(data, xvar, xlab) {
  ggplot(data, aes(x = .data[[xvar]], fill = graph_type, color = graph_type)) +
    geom_density(alpha = 0.35, linewidth = 0.7) +
    scale_fill_manual(values  = FILL_COLORS) +
    scale_color_manual(values = FILL_COLORS) +
    scale_x_log10(labels = comma) +
    labs(x = paste(xlab, "(log scale)"), y = "Density") +
    base_theme()
}

boxplot_compare <- function(data, yvar, ylab) {
  ggplot(data, aes(x = graph_type, y = .data[[yvar]], fill = graph_type)) +
    geom_violin(alpha = 0.4, trim = FALSE) +
    geom_boxplot(width = 0.18, outlier.shape = NA, alpha = 0.7) +
    scale_fill_manual(values = FILL_COLORS) +
    scale_y_log10(labels = comma) +
    labs(x = NULL, y = paste(ylab, "(log scale)")) +
    base_theme() +
    theme(legend.position = "none")
}

# ── nodes token distribution ──────────────────────────────────────────────────

p_nodes_density <- density_plot(df, "nodes_tokens", "Node tokens")
# Rscript has no screen: printing a plot would write a stray Rplots.pdf
if (interactive()) print(p_nodes_density)

p_nodes_box <- boxplot_compare(df, "nodes_tokens", "Node tokens")
if (interactive()) print(p_nodes_box)

# ── edges token distribution ──────────────────────────────────────────────────

p_edges_density <- density_plot(df, "edges_tokens", "Edge tokens")
if (interactive()) print(p_edges_density)

p_edges_box <- boxplot_compare(df, "edges_tokens", "Edge tokens")
if (interactive()) print(p_edges_box)

# ── graph tokens (nodes + edges) ──────────────────────────────────────────────

p_graph_density <- density_plot(df, "graph_tokens", "Graph tokens (nodes + edges)")
if (interactive()) print(p_graph_density)

# ── faceted: nodes + edges side by side ───────────────────────────────────────

df_long <- df %>%
  pivot_longer(
    cols      = c(nodes_tokens, edges_tokens),
    names_to  = "section",
    values_to = "tokens"
  ) %>%
  mutate(section = recode(section,
    "nodes_tokens" = "Nodes",
    "edges_tokens" = "Edges"
  ))

p_facet_density <- ggplot(
  df_long,
  aes(x = tokens, fill = graph_type, color = graph_type, linetype = graph_type)
) +
  geom_density(alpha = 0.35, linewidth = 0.7) +
  facet_wrap(~section, scales = "free") +
  scale_fill_manual(values  = FILL_COLORS) +
  scale_color_manual(values = FILL_COLORS) +
  scale_linetype_manual(values = LINE_TYPES) +
  scale_x_log10(labels = comma) +
  labs(x = "Tokens (log scale)", y = "Density") +
  base_theme()

if (interactive()) print(p_facet_density)

p_facet_box <- ggplot(
  df_long,
  aes(x = graph_type, y = tokens, fill = graph_type)
) +
  geom_violin(alpha = 0.4, trim = FALSE) +
  geom_boxplot(width = 0.18, outlier.shape = NA, alpha = 0.7) +
  facet_wrap(~section, scales = "free_y") +
  scale_fill_manual(values = FILL_COLORS) +
  scale_y_log10(labels = comma) +
  labs(x = NULL, y = "Tokens (log scale)") +
  base_theme() +
  theme(legend.position = "none")

if (interactive()) print(p_facet_box)

# ── compression ratio (Raw / Compressed) per repo ─────────────────────────────

df_ratio <- df %>%
  select(repo_id, graph_type, nodes_tokens, edges_tokens, graph_tokens,
         num_nodes, num_edges, node_compression_ratio, edge_compression_ratio) %>%
  pivot_wider(
    names_from  = graph_type,
    values_from = c(nodes_tokens, edges_tokens, graph_tokens, num_nodes, num_edges)
  ) %>%
  mutate(
    node_token_ratio  = nodes_tokens_Raw / nodes_tokens_Compressed,
    edge_token_ratio  = edges_tokens_Raw / edges_tokens_Compressed,
    graph_token_ratio = graph_tokens_Raw / graph_tokens_Compressed,
  ) %>%
  filter(
    is.finite(node_token_ratio),
    is.finite(graph_token_ratio)
  )

p_ratio_density <- df_ratio %>%
  pivot_longer(
    c(node_token_ratio, edge_token_ratio),
    names_to  = "section",
    values_to = "ratio"
  ) %>%
  mutate(section = recode(section,
    "node_token_ratio" = "Nodes",
    "edge_token_ratio" = "Edges"
  )) %>%
  filter(is.finite(ratio)) %>%
  ggplot(aes(x = ratio, fill = section, color = section)) +
  geom_density(alpha = 0.35, linewidth = 0.7) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "grey40") +
  facet_wrap(~section, scales = "free_y") +
  scale_fill_brewer(palette  = "Set2") +
  scale_color_brewer(palette = "Set2") +
  scale_x_log10(labels = comma) +
  labs(x = "Compression ratio — Raw / Compressed (log scale)", y = "Density") +
  base_theme() +
  theme(legend.position = "none")

if (interactive()) print(p_ratio_density)

# ── ratio vs raw size ─────────────────────────────────────────────────────────

df_ratio_long <- df_ratio %>%
  pivot_longer(
    cols      = c(node_token_ratio, edge_token_ratio),
    names_to  = "section",
    values_to = "ratio"
  ) %>%
  mutate(
    section  = recode(section,
      "node_token_ratio" = "Nodes",
      "edge_token_ratio" = "Edges"
    ),
    raw_size = if_else(section == "Nodes", num_nodes_Raw, num_edges_Raw)
  ) %>%
  filter(is.finite(ratio), is.finite(raw_size), raw_size > 0)

p_ratio_vs_size <- ggplot(
  df_ratio_long,
  aes(x = raw_size, y = ratio)
) +
  geom_hex(bins = 60) +
  geom_smooth(method = "lm", se = TRUE, color = "#E07B54", linewidth = 0.9) +
  facet_wrap(~section, scales = "free_x") +
  scale_x_log10(labels = comma) +
  scale_y_log10(labels = comma) +
  scale_fill_viridis_c(option = "magma", trans = "log10", labels = comma) +
  labs(
    x = "Raw graph size — node/edge count (log scale)",
    y = "Compression ratio — Raw / Compressed (log scale)",
    fill = "Repos"
  ) +
  base_theme()

if (interactive()) print(p_ratio_vs_size)

# ── prompt composition ────────────────────────────────────────────────────────

CTX_CATEGORIES <- c("readme", "docker", "dependencies", "build")
ctx_cols        <- paste0("ctx_", CTX_CATEGORIES, "_tokens")
ctx_labels      <- c("Context (Readme)", "Context (Docker)", "Context (Dependencies)", "Context (Build)")

df_comp <- df %>%
  filter(graph_type == "Compressed") %>%
  select(repo_id, nodes_tokens, edges_tokens,
         all_of(ctx_cols), package_files_tokens, total_tokens) %>%
  pivot_longer(
    cols      = c(nodes_tokens, edges_tokens, all_of(ctx_cols), package_files_tokens),
    names_to  = "section",
    values_to = "tokens"
  ) %>%
  mutate(
    section = recode(section,
      "nodes_tokens"              = "Nodes",
      "edges_tokens"              = "Edges",
      "ctx_readme_tokens"         = "Context (Readme)",
      "ctx_docker_tokens"         = "Context (Docker)",
      "ctx_dependencies_tokens"   = "Context (Dependencies)",
      "ctx_build_tokens"          = "Context (Build)",
      "package_files_tokens"      = "Package files"
    ),
    section = factor(section, levels = c(
      ctx_labels, "Nodes", "Edges", "Package files"
    )),
    fraction = tokens / total_tokens
  )

p_composition_box <- ggplot(df_comp, aes(x = section, y = fraction, fill = section)) +
  geom_violin(alpha = 0.4, trim = FALSE) +
  geom_boxplot(width = 0.18, outlier.shape = NA, alpha = 0.7) +
  scale_fill_brewer(palette = "Set2") +
  scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1)) +
  labs(x = NULL, y = "Fraction of total prompt tokens") +
  base_theme() +
  theme(legend.position = "none", axis.text.x = element_text(angle = 25, hjust = 1))

if (interactive()) print(p_composition_box)

df_comp_median <- df_comp %>%
  group_by(section) %>%
  summarise(median_fraction = median(fraction, na.rm = TRUE), .groups = "drop")

p_composition_bar <- ggplot(df_comp_median, aes(x = reorder(section, median_fraction), y = median_fraction, fill = section)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = percent(median_fraction, accuracy = 0.1)), hjust = -0.15, size = 3.5) +
  coord_flip() +
  scale_fill_brewer(palette = "Set2") +
  scale_y_continuous(labels = percent_format(accuracy = 1), expand = expansion(mult = c(0, 0.15))) +
  labs(x = NULL, y = "Median fraction of total prompt tokens") +
  base_theme() +
  theme(legend.position = "none")

if (interactive()) print(p_composition_bar)

# ── context docs breakdown ────────────────────────────────────────────────────

df_ctx <- df %>%
  filter(graph_type == "Compressed") %>%
  select(repo_id, total_tokens, context_files_tokens, all_of(ctx_cols)) %>%
  pivot_longer(
    cols      = all_of(ctx_cols),
    names_to  = "category",
    values_to = "tokens"
  ) %>%
  mutate(
    category = recode(category,
      "ctx_readme_tokens"       = "Readme",
      "ctx_docker_tokens"       = "Docker",
      "ctx_dependencies_tokens" = "Dependencies",
      "ctx_build_tokens"        = "Build"
    ),
    category            = factor(category, levels = c("Readme", "Docker", "Dependencies", "Build")),
    frac_of_total       = tokens / total_tokens,
    frac_of_context     = tokens / context_files_tokens
  )

# fraction of total tokens
p_ctx_frac_total <- df_ctx %>%
  group_by(category) %>%
  summarise(median_frac = median(frac_of_total, na.rm = TRUE), .groups = "drop") %>%
  ggplot(aes(x = reorder(category, median_frac), y = median_frac, fill = category)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = percent(median_frac, accuracy = 0.1)), hjust = -0.15, size = 3.5) +
  coord_flip() +
  scale_fill_brewer(palette = "Pastel1") +
  scale_y_continuous(labels = percent_format(accuracy = 1), expand = expansion(mult = c(0, 0.18))) +
  labs(x = NULL, y = "Median fraction of total prompt tokens") +
  base_theme() +
  theme(legend.position = "none")

if (interactive()) print(p_ctx_frac_total)

# fraction within context files
p_ctx_frac_within <- df_ctx %>%
  filter(is.finite(frac_of_context)) %>%
  group_by(category) %>%
  summarise(median_frac = median(frac_of_context, na.rm = TRUE), .groups = "drop") %>%
  ggplot(aes(x = reorder(category, median_frac), y = median_frac, fill = category)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = percent(median_frac, accuracy = 0.1)), hjust = -0.15, size = 3.5) +
  coord_flip() +
  scale_fill_brewer(palette = "Pastel1") +
  scale_y_continuous(labels = percent_format(accuracy = 1), expand = expansion(mult = c(0, 0.18))) +
  labs(x = NULL, y = "Median fraction of context file tokens") +
  base_theme() +
  theme(legend.position = "none")

if (interactive()) print(p_ctx_frac_within)

# distribution per category (fraction of total)
p_ctx_box <- ggplot(df_ctx, aes(x = category, y = frac_of_total, fill = category)) +
  geom_violin(alpha = 0.4, trim = FALSE) +
  geom_boxplot(width = 0.18, outlier.shape = NA, alpha = 0.7) +
  scale_fill_brewer(palette = "Pastel1") +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  labs(x = NULL, y = "Fraction of total prompt tokens") +
  base_theme() +
  theme(legend.position = "none")

if (interactive()) print(p_ctx_box)

# ── context window budget ─────────────────────────────────────────────────────

CONTEXT_LIMITS <- tibble(
  label  = c("32K", "64K", "128K", "256K"),
  tokens = c(32000L, 64000L, 128000L, 256000L)
)

df_budget <- df %>%
  filter(graph_type == "Compressed") %>%
  select(repo_id, total_tokens)

pct_fits <- CONTEXT_LIMITS %>%
  mutate(pct = map_dbl(tokens, ~ mean(df_budget$total_tokens <= .x, na.rm = TRUE)))

p_context_budget <- ggplot(df_budget, aes(x = total_tokens)) +
  geom_density(fill = "#4E8098", color = "#4E8098", alpha = 0.4, linewidth = 0.7) +
  geom_vline(
    data  = pct_fits,
    aes(xintercept = tokens),
    color = "#E07B54", linetype = "dashed", linewidth = 0.7
  ) +
  geom_label(
    data  = pct_fits,
    aes(x = tokens, y = Inf,
        label = paste0(label, "\n", percent(pct, accuracy = 0.1), " fit")),
    vjust = 1.2, hjust = 1.1,
    color = "#E07B54", fill = "white", size = 3, label.size = 0
  ) +
  scale_x_continuous(labels = comma) +
  labs(x = "Total prompt tokens", y = "Density") +
  base_theme()

if (interactive()) print(p_context_budget)
print(pct_fits)

# ── context budget: Raw vs Compressed overlay ─────────────────────────────────

df_budget_both <- df %>%
  select(repo_id, graph_type, total_tokens)

pct_fits_raw <- CONTEXT_LIMITS %>%
  mutate(
    graph_type = "Raw",
    pct        = map_dbl(tokens, ~ mean(df_budget_both$total_tokens[df_budget_both$graph_type == "Raw"] <= .x, na.rm = TRUE))
  )

pct_fits_compressed <- CONTEXT_LIMITS %>%
  mutate(
    graph_type = "Compressed",
    pct        = map_dbl(tokens, ~ mean(df_budget_both$total_tokens[df_budget_both$graph_type == "Compressed"] <= .x, na.rm = TRUE))
  )

p_context_budget_compare <- ggplot(
  df_budget_both,
  aes(x = total_tokens, fill = graph_type, color = graph_type, linetype = graph_type)
) +
  geom_density(alpha = 0.28, linewidth = 0.8) +
  geom_vline(
    data = CONTEXT_LIMITS,
    aes(xintercept = tokens),
    color = "grey35", linetype = "dashed", linewidth = 0.55
  ) +

  # Put labels in rows instead of stacked vertical text
  geom_label(
    data = CONTEXT_LIMITS,
    aes(x = tokens, y = 1.12, label = label),
    inherit.aes = FALSE,
    size = 2.45,
    label.size = 0, label.padding = unit(0.1, "lines"),
    color = "grey20"
  ) +
  geom_label(
    data = pct_fits_raw,
    aes(x = tokens, y = 1.00, label = percent(pct, accuracy = 1), color = graph_type),
    inherit.aes = FALSE,
    size = 2.45,
    label.size = 0, label.padding = unit(0.1, "lines"),
    show.legend = FALSE
  ) +
  geom_label(
    data = pct_fits_compressed,
    aes(x = tokens, y = 0.88, label = percent(pct, accuracy = 1), color = graph_type),
    inherit.aes = FALSE,
    size = 2.45,
    label.size = 0, label.padding = unit(0.1, "lines"),
    show.legend = FALSE
  ) +
  scale_fill_manual(values = FILL_COLORS) +
  scale_color_manual(values = FILL_COLORS) +
  scale_linetype_manual(values = LINE_TYPES) +
  scale_x_log10(
    labels = label_number(scale_cut = cut_short_scale()),
    breaks = c(1e3, 1e4, 1e5, 1e6),
    expand = expansion(mult = c(0.01, 0.01))
  ) +
  # the tails beyond are near zero; a narrower range leaves room for the labels
  coord_cartesian(xlim = c(300, 6e5)) +
  labs(x = "Total prompt tokens (log scale)", y = "Density") +
  base_theme() +
  theme(
    legend.position = "top",
    legend.title = element_blank(),
    # saved at the column width, so these are the printed sizes (pt)
    axis.title = element_text(size = 8.5),
    axis.text = element_text(size = 7.5),
    legend.text = element_text(size = 8.5),
    legend.key.size = unit(0.8, "lines"),
    legend.margin = margin(0, 0, 0, 0),
    legend.box.spacing = unit(0.1, "lines"),
    plot.margin = margin(0, 2, 0, 0)
  )


if (interactive()) print(p_context_budget_compare)

# ── compression ratio vs raw graph tokens ─────────────────────────────────────

df_ratio_vs_tokens <- df_ratio %>%
  filter(is.finite(graph_token_ratio), graph_tokens_Raw > 0)

p_ratio_vs_tokens <- ggplot(df_ratio_vs_tokens, aes(x = graph_tokens_Raw, y = graph_token_ratio)) +
  geom_hex(bins = 60) +
  geom_smooth(method = "lm", se = TRUE, color = "#E07B54", linewidth = 0.9) +
  scale_x_log10(labels = comma) +
  scale_y_log10(labels = comma) +
  scale_fill_viridis_c(option = "magma", trans = "log10", labels = comma) +
  labs(
    x    = "Raw graph tokens (log scale)",
    y    = "Compression ratio — Raw / Compressed (log scale)",
    fill = "Repos"
  ) +
  base_theme()

if (interactive()) print(p_ratio_vs_tokens)

# ── save ──────────────────────────────────────────────────────────────────────

plots_dir <- file.path(base_path, "figures", "plots")

save_plot <- function(filename, plot, width = 8, height = 4) {
  ggsave(file.path(plots_dir, paste0(filename, ".pdf")), plot, width = width, height = height)
  ggsave(file.path(plots_dir, paste0(filename, ".png")), plot, width = width, height = height, dpi = 300)
}

save_plot("token_stats_nodes_density",             p_nodes_density)
save_plot("token_stats_nodes_box",                 p_nodes_box)
save_plot("token_stats_edges_density",             p_edges_density)
save_plot("token_stats_edges_box",                 p_edges_box)
save_plot("token_stats_graph_density",             p_graph_density)
save_plot("token_stats_nodes_edges_facet_density", p_facet_density, 4, 3)
save_plot("token_stats_nodes_edges_facet_box",     p_facet_box)
save_plot("token_stats_compression_ratio",         p_ratio_density)
save_plot("token_stats_ratio_vs_size",             p_ratio_vs_size)
save_plot("token_stats_ratio_vs_tokens",           p_ratio_vs_tokens, 3,3)
save_plot("token_stats_composition_box",           p_composition_box, 8, 5)
save_plot("token_stats_composition_bar",           p_composition_bar)
save_plot("token_stats_context_budget",            p_context_budget)
# IEEE column width (3.5 in), so the paper shows it at 100%
save_plot("token_stats_context_budget_compare",    p_context_budget_compare, 3.5, 1.7)
save_plot("token_stats_ctx_frac_total",            p_ctx_frac_total, 6, 3.5)
save_plot("token_stats_ctx_frac_within",           p_ctx_frac_within, 6, 3.5)
save_plot("token_stats_ctx_box",                   p_ctx_box, 7, 4)
