#!/bin/sh
# Wraps the real mmdc binary with puppeteer config needed for Docker
exec /usr/local/bin/mmdc-real --puppeteerConfigFile /etc/mermaid-puppeteer.json "$@"
