import csv
import re

import scrapy

GITHUB_REPO_RE = re.compile(
    r"https?://(?:www\.)?github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)",
    re.IGNORECASE,
)


def extract_repo(html: str) -> str | None:
    """
    Return the first GitHub repo found in the HTML.
    """
    for m in GITHUB_REPO_RE.finditer(html):
        owner, repo = m.group(1), m.group(2)

        # remove .git if present
        if repo.lower().endswith(".git"):
            repo = repo[:-4]

        return f"https://github.com/{owner}/{repo}"

    return None


class GithubRepoSpider(scrapy.Spider):
    name = "github_repo"
    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        "DOWNLOAD_DELAY": 0.4,
        "AUTOTHROTTLE_ENABLED": True,
        "USER_AGENT": "repofinder/1.0",
        "LOG_LEVEL": "INFO",
    }

    def __init__(self, input_csv="data/input.csv", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_csv = input_csv

    def start_requests(self):
        with open(self.input_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = row.get("url")
                if not url:
                    continue

                yield scrapy.Request(
                    url=url,
                    callback=self.parse,
                    errback=self.on_error,
                    meta={"row": row},
                )

    def on_error(self, failure):
        # Do nothing: skip rows with no GitHub
        pass

    def parse(self, response):
        row = response.meta["row"]

        html = response.text
        repo = extract_repo(html)

        if not repo:
            return  # skip if no GitHub repo found

        yield {
            "name": row["name"],
            "url": repo,
            "awesome_descr": row["awesome_descr"],
            "language": row["language"],
        }
