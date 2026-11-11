#!/bin/sh
# Download the public-domain documents of the knowledge base (U.S. government works) and check
# they are byte-for-byte the files the project was built with: the analyst cites PAGE numbers, so a
# silently different edition would make its citations wrong. The owner's copyrighted textbook is
# never part of the image (it is not downloadable, and .dockerignore excludes every local PDF).
set -eu
mkdir -p docs/pdfs
fetch() {  # url  file  sha256
  curl -fsSL --retry 3 -A "traffic-analyst-build" -o "docs/pdfs/$2" "$1"
  echo "$3  docs/pdfs/$2" | sha256sum -c -
}
fetch https://ops.fhwa.dot.gov/publications/fhwahop08024/fhwa_hop_08_024.pdf \
      fhwa_signal_timing_manual_2008.pdf 675df7afec87adcef9a6218c14be35a58d153476e0a422a67d734b8634a6df7b
fetch https://mutcd.fhwa.dot.gov/pdfs/2009r1r2/part4.pdf \
      mutcd_2009_part4_signals.pdf 45fa0fdb9efb40b35848df40ce1a51260cad8033d7e538cfb5e983abac4996d9
