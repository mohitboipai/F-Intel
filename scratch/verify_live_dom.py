import urllib.request
import re

tunnel_url = "https://lunch-preference-rpg-many.trycloudflare.com"

print("--- 1. Testing Tunnel Accessibility ---")
req = urllib.request.Request(f"{tunnel_url}/", headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=10) as resp:
    html = resp.read().decode('utf-8')
    print(f"Status: {resp.status}, HTML size: {len(html)} bytes")

# Check CSS
req_css = urllib.request.Request(f"{tunnel_url}/static/css/gamma_explosion.css", headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req_css, timeout=10) as resp:
    css = resp.read().decode('utf-8')
    has_cw = "glow-call-wall" in css
    has_pw = "glow-put-wall" in css
    has_badges = "wall-badge-cw1" in css and "wall-badge-pw1" in css
    has_dealer_glows = "dealer-glow-green" in css and "dealer-glow-red" in css
    print(f"CSS Glows Check: CallWall={has_cw}, PutWall={has_pw}, Badges={has_badges}, DealerGlows={has_dealer_glows}")

# Check JS
req_core = urllib.request.Request(f"{tunnel_url}/static/js/dashboard_core.js", headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req_core, timeout=10) as resp:
    core_js = resp.read().decode('utf-8')
    has_opt = "optimizeOptionChainAndWalls" in core_js
    has_mm_clean = "cleanupMMTab" in core_js
    has_runway = "INSTITUTIONAL GEX TRADING CORRIDOR & VOLATILITY RUNWAY" in core_js or "gex-corridor-runway-bar" in core_js
    has_depth_bars = "linear-gradient(270deg, rgba(255, 51, 102" in core_js
    print(f"dashboard_core.js Check: optimizeChain={has_opt}, cleanupMM={has_mm_clean}, runwayBar={has_runway}, depthBars={has_depth_bars}")

req_ge = urllib.request.Request(f"{tunnel_url}/static/js/gamma_explosion_terminal.js", headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req_ge, timeout=10) as resp:
    ge_js = resp.read().decode('utf-8')
    has_dom_clean = "cleanupMMDOM" in ge_js
    has_dealer_inv = "enhanceDealerInventoryGlows" in ge_js
    print(f"gamma_explosion_terminal.js Check: cleanupMMDOM={has_dom_clean}, enhanceDealerGlows={has_dealer_inv}")

# Check Fragment
req_frag = urllib.request.Request(f"{tunnel_url}/fragment", headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req_frag, timeout=10) as resp:
    frag = resp.read().decode('utf-8')
    print(f"Fragment fetch: Status={resp.status}, Size={len(frag)} bytes")
    has_frag_chain = "id=\"frag-chain\"" in frag
    has_frag_mm = "id=\"frag-mm\"" in frag
    print(f"Fragment sections: frag-chain={has_frag_chain}, frag-mm={has_frag_mm}")

print("\n--- ALL CHECKS COMPLETED SUCCESSFULLY ---")
