#!/usr/bin/env Rscript
# Dump the current structure of the upstream R packages as JSON, for parity
# drift-checking against the Python port. Prints to stdout.
#
#   Rscript tools/parity_probe.R > /tmp/upstream.json
#
# Captures, per package: version, exported object names, and the dimensions +
# column names of every bundled data frame.

suppressMessages({
  library(jsonlite)
})

PACKAGES <- c("moderndive", "infer")

probe_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    return(NULL)
  }
  suppressMessages(library(pkg, character.only = TRUE))
  items <- tryCatch(data(package = pkg)$results[, "Item"], error = function(e) character(0))
  # `Item` can look like "name (alias)"; keep the leading token.
  items <- unique(sub("\\s.*$", "", items))
  datasets <- list()
  env <- new.env()
  for (nm in items) {
    ok <- tryCatch(
      {
        data(list = nm, package = pkg, envir = env)
        TRUE
      },
      error = function(e) FALSE
    )
    if (!ok || !exists(nm, envir = env)) next
    obj <- get(nm, envir = env)
    if (is.data.frame(obj)) {
      # I() keeps length-1 vectors as JSON arrays (auto_unbox would scalarize them).
      datasets[[nm]] <- list(rows = nrow(obj), cols = ncol(obj), columns = I(names(obj)))
    }
  }
  list(
    version = as.character(packageVersion(pkg)),
    exports = I(sort(getNamespaceExports(pkg))),
    datasets = datasets
  )
}

out <- list()
for (pkg in PACKAGES) {
  res <- probe_pkg(pkg)
  if (!is.null(res)) out[[pkg]] <- res
}
cat(toJSON(out, auto_unbox = TRUE, pretty = TRUE))
