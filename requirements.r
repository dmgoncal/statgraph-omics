# install_packages.R

# Install CRAN packages
install.packages("pacman", repos = "https://cloud.r-project.org")
install.packages("BiocManager", repos = "https://cloud.r-project.org")

# Use BiocManager to install Bioconductor packages
BiocManager::install("sesame", force = TRUE)
BiocManager::install("sesameData", force = TRUE)

# Install remotes (for GitHub installation)
install.packages("remotes", repos = "https://cloud.r-project.org")

# Install TCGAbiolinks from GitHub (devel branch)
remotes::install_github("BioinformaticsFMRP/TCGAbiolinks", ref = "devel", force = TRUE)
