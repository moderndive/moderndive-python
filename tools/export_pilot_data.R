# One-off: export pilot datasets for the Python port to CSV.
# Run from the ModernDive_book project root so renv libraries are active.
suppressMessages({
  library(moderndive)
  library(nycflights23)
  library(readr)
  library(dplyr)
})
have_gapminder <- requireNamespace("gapminder", quietly = TRUE)

outdir <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(outdir)) stop("Pass an output directory")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

dump <- function(df, name) {
  df <- as.data.frame(df)
  readr::write_csv(df, file.path(outdir, paste0(name, ".csv")))
  cat(sprintf("wrote %-28s %6d rows  %2d cols\n", name, nrow(df), ncol(df)))
}

# moderndive package datasets
dump(envoy_flights, "envoy_flights")
dump(early_january_2023_weather, "early_january_2023_weather")
dump(spotify_by_genre, "spotify_by_genre")
dump(almonds_sample_100, "almonds_sample_100")
dump(almonds_bowl, "almonds_bowl")
dump(un_member_states_2024, "un_member_states_2024")

# ISLR2 (multiple-regression Credit example in Ch 6)
if (requireNamespace("ISLR2", quietly = TRUE)) {
  data(Credit, package = "ISLR2")
  dump(Credit, "credit")
} else {
  cat("ISLR2 not available; skipping credit\n")
}

# nycflights23
dump(flights, "flights")
dump(weather, "weather")
dump(airlines, "airlines")
dump(airports, "airports")
dump(planes, "planes")

# fivethirtyeight (tidy-data chapter: drinks + airline_safety)
if (requireNamespace("fivethirtyeight", quietly = TRUE)) {
  dump(fivethirtyeight::drinks, "drinks")
  dump(fivethirtyeight::airline_safety, "airline_safety")
} else {
  cat("fivethirtyeight not available; skipping drinks/airline_safety\n")
}

# dem_score is a CSV mirror in the R book's data/ dir (no R package); copy it in
# separately when staging (cp data/dem_score.csv <outdir>/).

# gapminder (book builds gapminder_2007 by filtering year == 2007)
if (have_gapminder) {
  dump(gapminder::gapminder, "gapminder")
  dump(filter(gapminder::gapminder, year == 2007), "gapminder_2007")
} else {
  cat("gapminder package not available; will source via Python gapminder package\n")
}

cat("done\n")
