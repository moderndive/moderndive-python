#!/usr/bin/env Rscript
# Export the 25 extra datasets (24 moderndive + 1 infer) to CSV for the
# Python package build pipeline. Factors are coerced to character so they
# round-trip as strings through Parquet.

suppressMessages({
  library(moderndive)
  library(infer)
  library(readr)
})

out_dir <- "/tmp/extra_csv"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

moderndive_sets <- c(
  "DD_vs_SB", "MA_schools", "alaska_flights", "amazon_books", "avocados",
  "babies", "bowl_sample_1", "bowl_samples", "coffee_ratings",
  "early_january_weather", "ev_charging", "evals", "ipf_lifts",
  "ma_traffic_2020_vs_2019", "mario_kart_auction", "mass_traffic_2020",
  "orig_pennies_sample", "pennies", "pennies_resamples", "pennies_sample",
  "promotions", "promotions_shuffled", "spotify_52_original",
  "spotify_52_shuffled"
)

infer_sets <- c("gss")

coerce_factors <- function(df) {
  df <- as.data.frame(df)
  for (col in names(df)) {
    if (is.factor(df[[col]])) {
      df[[col]] <- as.character(df[[col]])
    }
  }
  df
}

export_one <- function(name, pkg) {
  data(list = name, package = pkg, envir = environment())
  df <- coerce_factors(get(name, envir = environment()))
  path <- file.path(out_dir, paste0(name, ".csv"))
  readr::write_csv(df, path, na = "NA")
  cat(sprintf("%-28s %6d rows %3d cols  (%s)\n", name, nrow(df), ncol(df), pkg))
}

for (nm in moderndive_sets) export_one(nm, "moderndive")
for (nm in infer_sets) export_one(nm, "infer")

cat(sprintf("\nWrote %d CSVs to %s\n",
            length(moderndive_sets) + length(infer_sets), out_dir))
