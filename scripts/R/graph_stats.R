library(tidyverse)
library(lubridate)
library(scales)
library(this.path)

script_dir <- dirname(this.path())
base_path <- file.path(script_dir, "..", "..")

df <- readr::read_csv(file.path(base_path, "data", "interim", "dataset_stats.csv"))

df <- df %>%
  mutate(
    log_nodes = log10(nodes + 1),
    log_edges = log10(edges + 1),
    total_tokens = n_nodes_tokens + n_edges_tokens + n_package_files_tokens,
    log_total_tokens = log10(total_tokens + 1),

    density = if_else(
      nodes > 1,
      edges / (nodes * (nodes - 1)),
      NA_real_
    )
  )

lang_df <- readr::read_tsv(
  file.path(base_path, "data", "raw", "filter_50_2021_list.tsv"),
  col_names = c("repo", "language")
)

df <- df %>%
  left_join(lang_df, by = "repo")

plot_hist_with_percentiles <- function(data,
                                       xvar,
                                       xlab,
                                       bins = 100,
                                       probs = c(0.01, 0.05, 0.10, 0.90, 0.95, 0.99),
                                       facet_by_language = FALSE) {

  xq <- data %>%
    group_by(package_only) %>%
    summarise(
      q = list(stats::quantile(.data[[xvar]], probs = probs, na.rm = TRUE)),
      .groups = "drop"
    ) %>%
    unnest_wider(q) %>%
    pivot_longer(
      cols = -package_only,
      names_to = "perc",
      values_to = "x"
    )

  xq <- xq %>%
    mutate(
      perc_num = as.numeric(str_remove(perc, "%")),
      label = paste0("p", perc_num)
    )

  # Determine histogram max height (for label placement)
  ymax <- ggplot_build(
    ggplot(data, aes(x = .data[[xvar]])) +
      geom_histogram(bins = bins)
  )$data[[1]]$count %>%
    max(na.rm = TRUE)

  base <- ggplot(
    data,
    aes(x = .data[[xvar]], fill = package_only)
  ) +
    geom_histogram(bins = bins, position = "identity", alpha = 0.35) +
    geom_vline(
      data = xq,
      aes(xintercept = x, color = package_only, linetype = label),
      inherit.aes = FALSE,
      linewidth = 0.6
    ) +
    geom_text(
      data = xq,
      aes(x = x, y = ymax * 1.02, label = label, color = package_only),
      inherit.aes = FALSE,
      angle = 90,
      vjust = -0.3,
      size = 3
    ) +
    labs(
      x = xlab,
      y = "Count",
      fill = "package_only",
      color = "package_only",
      linetype = "Percentile"
    ) +
    theme_minimal()

  if (facet_by_language) {
    base <- base + facet_wrap(~ language, scales = "free_y")
  }

  trim_stats <- data %>%
    group_by(package_only) %>%
    summarise(
      p05 = quantile(.data[[xvar]], 0.05, na.rm = TRUE),
      p95 = quantile(.data[[xvar]], 0.95, na.rm = TRUE),
      n_total = n(),
      n_below_5 = sum(.data[[xvar]] < p05, na.rm = TRUE),
      n_above_95 = sum(.data[[xvar]] > p95, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      pct_below_5 = n_below_5 / n_total,
      pct_above_95 = n_above_95 / n_total,
      pct_total_trimmed = pct_below_5 + pct_above_95
    )

  print(trim_stats)

  base
}

p1 <- ggplot(df, aes(x = log_nodes, color = package_only)) +
  geom_density() +
  facet_wrap(~ language, scales = "free_y") +
  labs(x = "log10(nodes)")

# Rscript has no screen: printing a plot would write a stray Rplots.pdf
if (interactive()) print(p1)

p2 <- ggplot(df, aes(x = log_edges, color = package_only)) +
  geom_density() +
  facet_wrap(~ language, scales = "free_y") +
  labs(x = "log10(edges)")

if (interactive()) print(p2)


p_nodes_bar <- plot_hist_with_percentiles(
  data = df %>% filter(!is.na(log_nodes)),
  xvar = "log_nodes",
  xlab = "log10(nodes + 1)",
  bins = 100,
  probs = c(0.01, 0.05, 0.10, 0.90, 0.95, 0.99)
)

if (interactive()) print(p_nodes_bar)

p_edges_bar <- plot_hist_with_percentiles(
  data = df %>% filter(!is.na(log_edges)),
  xvar = "log_edges",
  xlab = "log10(edges + 1)",
  bins = 100,
  probs = c(0.01, 0.05, 0.10, 0.90, 0.95, 0.99)
)

if (interactive()) print(p_edges_bar)

p_lang_count <- ggplot(
  df %>%
    filter(!is.na(language)) %>%
    count(language),
  aes(x = reorder(language, n), y = n, fill = language)
) +
  geom_col() +
  coord_flip() +
  labs(
    x = "Language",
    y = "Repository Count"
  )
if (interactive()) print(p_lang_count)


p3 <- ggplot(df, aes(x = log_total_tokens, color = package_only)) +
  geom_density() +
  facet_wrap(~ language, scales = "free_y") +
  labs(x = "log10(total_tokens)")

if (interactive()) print(p3)


df_lang_tokens <- df %>%
  filter(!is.na(language)) %>%
  group_by(language) %>%
  summarise(total_tokens = sum(total_tokens, na.rm = TRUE)) %>%
  arrange(desc(total_tokens))

p_lang_tokens <- ggplot(df_lang_tokens,
       aes(x = reorder(language, total_tokens),
           y = total_tokens,
           fill = language)) +
  geom_col() +
  coord_flip() +
  scale_y_continuous(labels = scales::comma) +
  labs(
    x = "Language",
    y = "Total Tokens (Sum Across Repos)"
  )
if (interactive()) print(p_lang_tokens)


p4 <- ggplot(df, aes(x = density, color = package_only)) +
  geom_density() +
  facet_wrap(~ language, scales = "free_y") +
  labs(x = "graph density")

if (interactive()) print(p4)
