library(dplyr)
library(ggplot2)
library(readr)
library(ggforce)
library(ggbeeswarm)
library(stringr)
library(this.path)
script_dir <- dirname(this.path())
base_path <- file.path(script_dir, "..", "..")
docs_tok <- read_csv(
  file.path(base_path, "data", "interim", "repo_context_files_analyzed_only_stats.csv"),
  show_col_types = FALSE
)

glimpse(docs_tok)



docs_tok <- docs_tok %>%
  mutate(
    token_n = as.numeric(token_n),
    category = case_when(
      category == "readme" ~ "README",
      TRUE ~ str_to_title(category)
    ),
    category = as.factor(category),
    language = as.factor(language)
  )

# COUNTS PER REPO x CATEGORY
repo_cat_counts <- docs_tok %>%
  count(repo, category, name = "n_docs")

# SUMMARY FOR PANEL BARS
repo_cat_summary <- repo_cat_counts %>%
  group_by(category) %>%
  summarise(
    mean_docs = mean(n_docs, na.rm = TRUE),
    q25 = quantile(n_docs, 0.25, na.rm = TRUE),
    med = median(n_docs, na.rm = TRUE),
    q75 = quantile(n_docs, 0.75, na.rm = TRUE),
    n_repos = n(),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_docs))

repo_cat_summary <- repo_cat_counts %>%
  group_by(category) %>%
  summarise(
    mean_docs = mean(n_docs, na.rm = TRUE),
    sd_docs   = sd(n_docs, na.rm = TRUE),
    n         = n(),
    se        = sd_docs / sqrt(n),
    .groups = "drop"
  )

p1 <- ggplot(repo_cat_summary, aes(x = category, y = mean_docs, fill = category)) +
  geom_col(width = 0.6, show.legend = FALSE) +
  geom_errorbar(
    aes(ymin = mean_docs - se, ymax = mean_docs + se),
    width = 0.2
  ) +
  coord_flip() +
  scale_fill_brewer(palette = "Set3") +
  labs(
    x = "Category",
    y = "# Docs per repo"
  )+

  theme(
    legend.margin = margin(0, 0, 0, 0)
  )

# Rscript has no screen: printing a plot would write a stray Rplots.pdf
if (interactive()) print(p1)

ggsave(
  file.path(base_path, "figures", "plots", "context_docs_stats.pdf"),
  p1,
  width = 5,
  height = 3
)


docs_tok <- read_csv(
  file.path(base_path, "data", "interim", "repo_context_files_analyzed_only_stats.csv"),
  show_col_types = FALSE
)

docs_tok <- docs_tok %>%
  mutate(
    token_n = as.numeric(token_n),
    category = as.factor(category),
    language = as.factor(language)
  )

repo_cat_lang_counts <- docs_tok %>%
  count(repo, language, category, name = "n_docs")

repo_cat_summary <- repo_cat_lang_counts %>%
  group_by(language, category) %>%
  summarise(
    mean_docs = mean(n_docs, na.rm = TRUE),
    sd_docs = sd(n_docs, na.rm = TRUE),
    n = n(),
    se = sd_docs / sqrt(n),
    .groups = "drop"
  )

p2 <- ggplot(repo_cat_summary, aes(x = category, y = mean_docs, fill = category)) +
  geom_col(width = 0.6, show.legend = FALSE) +
  geom_errorbar(
    aes(ymin = mean_docs - se, ymax = mean_docs + se),
    width = 0.2
  ) +
  coord_flip() +
  facet_wrap(~ language, scales = "free_y") +
  scale_fill_brewer(palette = "Set3") +
  labs(
    title = "Average #docs per repo by category and language",
    x = "Category",
    y = "# docs per repo"
  )

if (interactive()) print(p2)


p1 <- ggplot(docs_tok, aes(x = token_n)) +
  geom_density() +
  labs(
    title = "Token count distribution across documents",
    x = "Tokens per document",
    y = "Density"
  )

if (interactive()) print(p1)


p_bounds <- quantile(docs_tok$token_n, probs = c(0.05, 0.95), na.rm = TRUE)

p5  <- p_bounds[[1]]
p95 <- p_bounds[[2]]

print(paste0("5th percentile: ", round(p5, 2)))
print(paste0("95th percentile: ", round(p95, 2)))

docs_tok_trimmed <- docs_tok %>%
  filter(token_n >= p5,
         token_n <= p95)

p2 <- ggplot(docs_tok_trimmed, aes(x = token_n)) +
  geom_density(fill = "steelblue", alpha = 0.4) +
  labs(
    title = "Token count distribution (5–95 percentile range)",
    subtitle = paste0("Range: ", round(p5, 1), " – ", round(p95, 1)),
    x = "Tokens per document",
    y = "Density"
  )

if (interactive()) print(p2)



p_docs_trim <- ggplot() +
  geom_density(data = docs_tok, aes(x = token_n),
               color = "red", alpha = 0.3) +
  geom_density(data = docs_tok_trimmed, aes(x = token_n),
               fill = "steelblue", alpha = 0.4) +
  labs(
    title = "Original vs Trimmed Distribution",
    x = "Tokens per document",
    y = "Density"
  )
if (interactive()) print(p_docs_trim)



p2 <- ggplot(docs_tok_trimmed, aes(x = category, y = token_n, fill = category)) +
  geom_violin(trim = FALSE, alpha = 0.75) +
  geom_boxplot(width = 0.18, outlier.shape = NA, alpha = 0.45) +
  coord_flip() +
  scale_fill_brewer(palette = "Set3") +
  labs(
    title = "Token count distribution by document category",
    x = "Category",
    y = "Tokens per document"
  ) +
  theme(legend.position = "none")

if (interactive()) print(p2)


p3 <- ggplot(docs_tok_trimmed, aes(x = category, y = token_n, fill = category)) +
  geom_violin(trim = FALSE, alpha = 0.75) +
  geom_boxplot(width = 0.18, outlier.shape = NA, alpha = 0.45) +
  coord_flip() +
  facet_wrap(~ language, scales = "free_y") +
  scale_fill_brewer(palette = "Set3") +
  labs(
    title = "Token count distribution by category, split by language",
    x = "Category",
    y = "Tokens per document"
  ) +
  theme(legend.position = "none")

if (interactive()) print(p3)


repo_col <- "repo"   # e.g. "repo_name" / "full_name" / "repo_id"

stopifnot(repo_col %in% names(docs_tok))

repo_ctx <- docs_tok %>%
  group_by(.data[[repo_col]]) %>%
  summarise(
    context_token_n = sum(token_n, na.rm = TRUE),
    doc_n = dplyr::n(),
    .groups = "drop"
  )


p_ctx1 <- ggplot(repo_ctx, aes(x = context_token_n)) +
  geom_density() +
  labs(
    title = "Repo context token distribution (sum of docs per repo)",
    x = "Tokens per repo (context_token_n)",
    y = "Density"
  )

if (interactive()) print(p_ctx1)


ctx_bounds <- quantile(repo_ctx$context_token_n, probs = c(0.05, 0.95), na.rm = TRUE)
ctx_p5  <- ctx_bounds[[1]]
ctx_p95 <- ctx_bounds[[2]]

print(paste0("Repo context 5th percentile: ", round(ctx_p5, 2)))
print(paste0("Repo context 95th percentile: ", round(ctx_p95, 2)))


repo_ctx_trimmed <- repo_ctx %>%
  filter(context_token_n >= ctx_p5,
         context_token_n <= ctx_p95)


p_ctx2 <- ggplot(repo_ctx_trimmed, aes(x = context_token_n)) +
  geom_density(fill = "darkgreen", alpha = 0.35) +
  labs(
    title = "Repo context token distribution (5–95 percentile range)",
    subtitle = paste0("Range: ", round(ctx_p5, 1), " – ", round(ctx_p95, 1)),
    x = "Tokens per repo (context_token_n)",
    y = "Density"
  )

if (interactive()) print(p_ctx2)


p_ctx_trim <- ggplot() +
  geom_density(data = repo_ctx, aes(x = context_token_n),
               color = "red", alpha = 0.3) +
  geom_density(data = repo_ctx_trimmed, aes(x = context_token_n),
               fill = "darkgreen", alpha = 0.35) +
  labs(
    title = "Repo context distribution: original vs trimmed",
    x = "Tokens per repo (context_token_n)",
    y = "Density"
  )
if (interactive()) print(p_ctx_trim)
