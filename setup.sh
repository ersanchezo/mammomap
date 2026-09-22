#!/usr/bin/env bash
# PathVision Tier 2 setup — run once on your laptop
# Usage: bash setup.sh

set -e
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓ $1${NC}"; }
info() { echo -e "${YELLOW}→ $1${NC}"; }
err()  { echo -e "${RED}✗ $1${NC}"; exit 1; }

echo ""
echo "╔══════════════════════════════════════╗"
echo "║  PathVision Tier 2 — Setup Script   ║"
echo "╚══════════════════════════════════════╝"
echo ""

# ── 0. Prerequisites check ────────────────────────────────────────────────────
info "Checking prerequisites…"
command -v python3 &>/dev/null || err "Python 3.10+ required (brew install python@3.11)"
command -v node    &>/dev/null || err "Node 18+ required (brew install node)"
command -v npm     &>/dev/null || err "npm required"
command -v git     &>/dev/null || err "git required"
ok "Prerequisites OK"

# ── 1. Modal ──────────────────────────────────────────────────────────────────
echo ""
info "Step 1/5 — Modal.com (serverless GPU)"
echo "  1. Go to https://modal.com and create a free account"
echo "  2. Press Enter when ready…"
read -r

pip install modal -q
modal setup   # opens browser for auth
ok "Modal authenticated"

# ── 2. Supabase ───────────────────────────────────────────────────────────────
echo ""
info "Step 2/5 — Supabase (database + storage)"
echo "  1. Go to https://supabase.com → New project"
echo "  2. Note your project URL and service-role key (Settings → API)"
echo "  3. Run supabase_schema.sql in the SQL editor"
echo "  4. Create a storage bucket named 'pathvision' (Storage → New bucket)"
echo ""
read -rp "  Supabase URL (https://xxxx.supabase.co): " SUPABASE_URL
read -rp "  Service role key: " SUPABASE_KEY
ok "Supabase credentials saved"

# ── 3. Upstash Redis ──────────────────────────────────────────────────────────
echo ""
info "Step 3/5 — Upstash Redis (free tier)"
echo "  1. Go to https://upstash.com → Create database"
echo "  2. Choose 'Global' region, free tier is fine"
echo "  3. Copy the Redis URL from the database details page"
echo ""
read -rp "  Redis URL (redis://default:xxx@xxx.upstash.io): " REDIS_URL
ok "Redis URL saved"

# ── 4. Modal secrets ─────────────────────────────────────────────────────────
echo ""
info "Step 4/5 — Uploading secrets to Modal"
modal secret create pathvision-secrets \
    SUPABASE_URL="$SUPABASE_URL" \
    SUPABASE_SERVICE_KEY="$SUPABASE_KEY" \
    REDIS_URL="$REDIS_URL"
ok "Secrets uploaded to Modal"

# ── 5. Deploy backend ─────────────────────────────────────────────────────────
echo ""
info "Step 5/5 — Deploying backend to Modal"
info "This builds the container image (~3 min on first run, cached after)"
modal deploy pathvision_modal.py

MODAL_URL=$(modal app list --json 2>/dev/null | python3 -c "
import json,sys
try:
    apps = json.load(sys.stdin)
    url = next((a.get('url','') for a in apps if a.get('name')=='pathvision'), '')
    print(url)
except: print('')
" 2>/dev/null)

ok "Backend deployed!"
echo ""
echo "  Backend URL: ${GREEN}${MODAL_URL:-https://YOUR-APP.modal.run}${NC}"
echo ""

# ── 6. Frontend env ───────────────────────────────────────────────────────────
if [ -d "frontend" ]; then
    echo "VITE_API_URL=${MODAL_URL}" > frontend/.env.local
    ok "Written frontend/.env.local"
fi

# ── 7. Vercel deploy ─────────────────────────────────────────────────────────
echo ""
info "Deploy frontend to Vercel?"
read -rp "  (y/n): " DEPLOY_VERCEL

if [ "$DEPLOY_VERCEL" = "y" ]; then
    npm i -g vercel -q
    echo ""
    echo "  Set this environment variable in Vercel:"
    echo "    VITE_API_URL = ${MODAL_URL}"
    echo ""
    cd frontend && vercel --prod
    cd ..
    ok "Frontend deployed to Vercel"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Setup complete! Summary                            ║"
echo "╠══════════════════════════════════════════════════════╣"
printf "║  Backend  %-43s║\n" "${MODAL_URL}"
printf "║  Supabase %-43s║\n" "${SUPABASE_URL}"
printf "║  Redis    %-43s║\n" "Upstash free tier"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Useful commands:                                   ║"
echo "║    modal logs pathvision          # live logs       ║"
echo "║    modal app list                 # see endpoints   ║"
echo "║    modal run pathvision_modal.py  # local test      ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
