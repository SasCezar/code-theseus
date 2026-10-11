.PHONY: help collect-urls dedupe fetch-metadata filter prefetch arcan context combine-dataset pipeline render-prompts pdf-docs

# Overridable on the command line, e.g. `make arcan TSV=data/raw/filter_100_2022_list.tsv`
TSV        ?= data/raw/filter_50_2021_list.tsv
DATASET    ?= data/out/reverse-se_v0.6_p10-95_ctx95_dataset_sample_1000.jsonl
VERSION    ?= v0.6

help:
	@echo "Dataset-build pipeline, in order (each stage's default input is the"
	@echo "previous stage's default output — see scripts/README.md):"
	@echo "  make collect-urls    scrape repo URLs                 -> data/raw/*_github.csv"
	@echo "  make dedupe          merge + dedupe those URLs        -> data/raw/full_list_deduped.csv"
	@echo "  make fetch-metadata  fetch GitHub metadata per url    -> data/raw/full_list_info.jsonl"
	@echo "  make filter          filter by language/stars/activity -> data/raw/filter_<stars>_<year>_list.tsv"
	@echo "  make prefetch        clone repos from the filter TSV  -> data/repository/"
	@echo "  make arcan           run Arcan dependency analysis    -> data/raw/arcanOutput/"
	@echo "  make context         scan README/build/docker/deps    -> data/interim/repo_context_files_all.jsonl"
	@echo "  make combine-dataset join arcan + context + metadata  -> the final dataset JSONL"
	@echo "  make pipeline        run all of the above, in order"
	@echo ""
	@echo "Independent of the above (generation/evaluation live in src/reverse_se, see their --help):"
	@echo "  make render-prompts DATASET=...  dump rendered prompts to .txt, no model call"
	@echo "  make pdf-docs                     render generated C4 docs to PDF"
	@echo ""
	@echo "Not wired here (run manually, see scripts/python/repofinder/): the Scrapy"
	@echo "spider that resolves non-GitHub URLs collected by collect-urls."

collect-urls:
	uv run scripts/python/extract_jgurry_urls.py
	uv run scripts/python/extract_awesome_urls.py
	uv run scripts/python/gitranking2csv.py

dedupe:
	uv run scripts/python/deduplicate.py

fetch-metadata:
	uv run scripts/python/fetch_github_info.py

filter:
	uv run scripts/python/filter_dataset.py

prefetch:
	./scripts/bash/prefetch_repo.sh $(TSV)

arcan:
	./scripts/bash/run_arcan_docker.sh $(TSV)

context:
	uv run scripts/python/context_file_extractor.py data/repository \
		--language-tsv $(TSV) \
		--out data/interim/repo_context_files_all.jsonl

combine-dataset:
	uv run scripts/python/combine_arcan_w_context.py \
		--arcanout_dir data/raw/arcanOutput \
		--context_file data/interim/repo_context_files_all.jsonl \
		--info_file data/raw/full_list_info.csv \
		--version $(VERSION)

pipeline: collect-urls dedupe fetch-metadata filter prefetch arcan context combine-dataset

render-prompts:
	uv run scripts/python/render_prompts.py --input_file $(DATASET) --output_dir data/generated_docs/prompts_out

pdf-docs:
	./scripts/bash/render_c4_pdf.sh
