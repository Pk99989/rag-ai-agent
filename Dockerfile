# SUPERSEDED (Phase 14): this single-image, Streamlit-imports-everything-
# directly setup was replaced when the FastAPI backend (Phase 12) split the
# app into two independent services. This file is kept in place only
# because the sandbox that made this edit couldn't delete it (a filesystem
# permission quirk on the mounted OneDrive folder) -- it is intentionally
# broken as a build target so nobody accidentally deploys the stale
# architecture by running `docker build .` without `-f`.
#
# Use instead:
#   docker compose up --build          (builds and runs both services)
#   docker build -f Dockerfile.api .   (FastAPI backend only)
#   docker build -f Dockerfile.ui .    (Streamlit UI only)
#
# You may delete this file yourself: `Remove-Item Dockerfile` (PowerShell).
RUN echo "This Dockerfile is superseded -- see the comment at the top of this file." && exit 1
