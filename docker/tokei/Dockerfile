# ---- build tokei from source (reliable across distros) ----
FROM rust:1.93-alpine AS builder

RUN apk add --no-cache musl-dev openssl-dev pkgconfig build-base git
RUN cargo install tokei --locked

# ---- runtime image ----
FROM alpine:3.23

RUN apk add --no-cache python3 py3-pip ca-certificates py3-tqdm

# Copy tokei binary from builder
COPY --from=builder /usr/local/cargo/bin/tokei /usr/local/bin/tokei

# Copy analyzer script
COPY analyze.py /usr/local/bin/analyze.py
RUN chmod +x /usr/local/bin/analyze.py

# Defaults:
#   input repos dir: /repos
#   output TSV: /out/results.tsv
ENV INPUT_DIR=/repos
ENV OUTPUT_TSV=/out/results.tsv

ENTRYPOINT ["python3", "/usr/local/bin/analyze.py"]
