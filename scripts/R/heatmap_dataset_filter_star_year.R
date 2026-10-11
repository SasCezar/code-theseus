library(tidyverse)
library(lubridate)
library(scales)
library(this.path)
library(metR)

script_dir <- dirname(this.path())
base_path <- file.path(script_dir, "..", "..")

# --- Load / clean ---
df <- readr::read_csv(
  file.path(base_path, "data", "raw", "full_list_info.csv"),
  show_col_types = FALSE
) %>%
  mutate(
    pushed_at = ymd_hms(pushed_at, tz = "UTC"),
    pushed_year = year(pushed_at),
    archived = as.logical(archived),
    stargazers_count = as.integer(stargazers_count)
  ) %>%
  filter(!is.na(stargazers_count), !is.na(pushed_year)) %>%
  filter(!archived)

n_total <- nrow(df)

star_grid <- seq(0, 300, by = 10)
year_grid <- 2015:year(Sys.Date())

combined <- tidyr::expand_grid(
  star_threshold = star_grid,
  year_threshold = year_grid
) %>%
  rowwise() %>%
  mutate(
    repos_remaining = sum(
      df$stargazers_count >= star_threshold &
        df$pushed_year >= year_threshold
    ),
    keep_frac = repos_remaining / n_total
  ) %>%
  ungroup()

pick_star <- 50
pick_year <- 2021

pick_stats <- combined %>%
  filter(star_threshold == pick_star, year_threshold == pick_year)

cont_levels <- c(0.3, 0.4, 0.5, 0.6, 0.7)

p1 <- ggplot(combined, aes(x = star_threshold, y = year_threshold)) +
  geom_tile(aes(fill = keep_frac)) +

  geom_contour(
    aes(z = keep_frac),
    breaks = cont_levels,
    color = "white",
    linewidth = 0.7
  ) +

  geom_text_contour(
    aes(z = keep_frac),
    breaks = cont_levels,
    size = 3.5,
    stroke = 0.1,
    color = "black"
  ) +

  geom_point(
    data = pick_stats,
    aes(x = star_threshold, y = year_threshold),
    inherit.aes = FALSE,
    shape = 18,
    size = 4,
    color = "black"
  ) +

  scale_fill_gradient(
    low = "#d6e9f7",
    high = "#08306b",
    labels = percent,
    name = "Coverage"
  ) +

  # 🔧 FIX: remove extra padding around plot
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(
    labels = function(x) as.integer(x),
    expand = c(0, 0)
  ) +

  labs(
    x = "Minimum stars",
    y = "Minimum last push year"
  ) +

  theme(
    panel.grid = element_blank(),
    panel.border = element_blank(),
    panel.background = element_blank(),
    plot.background = element_blank(),
    legend.margin = margin(0, 0, 0, 0)
  )

ggsave(
  file.path(base_path, "figures", "plots", "dataset_coverage_heatmap.pdf"),
  p1,
  width = 5,
  height = 3
)

ggsave(
  file.path(base_path, "figures", "plots", "dataset_coverage_heatmap.png"),
  p1,
  width = 5,
  height = 3,
  dpi = 300
)
