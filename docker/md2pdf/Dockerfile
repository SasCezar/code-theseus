FROM pandoc/latex:latest

USER root

# Install Node.js, Chromium, and fonts needed for mermaid rendering
RUN apk add --no-cache \
    nodejs \
    npm \
    chromium \
    nss \
    freetype \
    harfbuzz \
    ca-certificates \
    ttf-freefont \
    font-noto \
    fontconfig \
    && fc-cache -f

# Pre-install TinyTeX packages required by pandoc's default LaTeX template
RUN tlmgr update --self 2>/dev/null | true && \
    tlmgr install \
        collection-fontsrecommended \
        unicode-math \
        lm-math \
        listings \
        fancyvrb \
        caption \
    2>/dev/null | true

# Tell Puppeteer to use the system Chromium instead of downloading its own
ENV PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=true \
    PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium-browser

# Install mermaid CLI, then rename the real binary so we can wrap it
RUN npm install -g @mermaid-js/mermaid-cli \
    && mv /usr/local/bin/mmdc /usr/local/bin/mmdc-real

# Puppeteer launch config: disables sandbox (required in Docker)
COPY puppeteer-config.json /etc/mermaid-puppeteer.json

# mmdc wrapper — injects --puppeteerConfigFile into every mmdc call
COPY mmdc-wrapper.sh /usr/local/bin/mmdc
RUN chmod +x /usr/local/bin/mmdc

# diagram.lua — pandoc Lua filter for rendering mermaid (and other diagram types)
# Pulled from https://github.com/pandoc-ext/diagram at build time
ARG DIAGRAM_LUA_REF=main
RUN wget -q \
    "https://raw.githubusercontent.com/pandoc-ext/diagram/${DIAGRAM_LUA_REF}/_extensions/diagram/diagram.lua" \
    -O /usr/local/share/diagram.lua

# Single-file entrypoint
COPY render.sh /usr/local/bin/render
# Batch entrypoints (run inside one container for all files)
COPY render_one.sh      /usr/local/bin/render_one
COPY render_batch.sh    /usr/local/bin/render_batch
COPY batch_mermaid.js   /usr/local/bin/batch_mermaid.js
RUN chmod +x /usr/local/bin/render \
             /usr/local/bin/render_one \
             /usr/local/bin/render_batch \
             /usr/local/bin/batch_mermaid.js

WORKDIR /data

# Default: single-file mode. Override with --entrypoint render_batch for batch.
ENTRYPOINT ["render"]
