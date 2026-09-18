import re
import os

def clean_html(content: str) -> str:
    # 1. Clean up top-verdict-pill and fix brand-section
    # Ensure brand-section is cleanly closed
    content = re.sub(
        r'(<span class="brand-badge">PRO QUANT</span>)[\s\S]*?(</div>\s*</div>\s*<div class="market-strip">)',
        r'\1\n        </div>\n\n        <div class="market-strip">',
        content
    )
    # Also remove if single closing div
    content = re.sub(
        r'\s*<div id="top-verdict-pill"[\s\S]*?</div>',
        '',
        content,
        flags=re.IGNORECASE
    )

    # 2. Remove Theme toggle wrap
    content = re.sub(
        r'\s*<div class="theme-toggle-wrap"[\s\S]*?</div>',
        '',
        content,
        flags=re.IGNORECASE
    )

    # 3. Clean up Gamma tab button
    content = re.sub(
        r'<button class="tab-btn mm-tab-btn" data-tab="mm">[\s\S]*?</button>',
        '<button class="tab-btn mm-tab-btn" data-tab="mm"><span>GAMMA EXPLOSION & MM</span></button>',
        content
    )

    # 4. Remove frag-verdict-transfer if present
    content = re.sub(
        r'\s*<span id="frag-verdict-transfer"[\s\S]*?</span>',
        '',
        content
    )

    # 5. Restyle legacy baby blue #4fc3f7 to IV Surface Electric Cyan #00e5ff
    content = content.replace('#4fc3f7', '#00e5ff')
    content = content.replace('#4FC3F7', '#00e5ff')

    # 6. Replace clashing legacy backgrounds with IV Surface dark slate
    content = content.replace(
        'background:linear-gradient(135deg, rgba(18,18,42,0.95), rgba(10,14,28,0.95));',
        'background:var(--bg-surface, #1e222d); border: 1px solid var(--border-card, #363c4e);'
    )
    content = content.replace('background:rgba(18,18,42,0.85);', 'background:rgba(30, 34, 45, 0.7);')
    content = content.replace('background:rgba(18,18,42,0.95);', 'background:var(--bg-surface, #1e222d);')
    content = content.replace('background:rgba(14,18,36,0.9);', 'background:rgba(30, 34, 45, 0.7);')
    content = content.replace('background:rgba(22,20,38,0.9);', 'background:rgba(30, 34, 45, 0.7);')
    content = content.replace('background:rgba(18,14,36,0.9);', 'background:rgba(30, 34, 45, 0.7);')
    content = content.replace('background:#0d1124;', 'background:#131722;')
    content = content.replace('background:#0e1022;', 'background:#131722;')
    content = content.replace('background:#0a0d1e;', 'background:#131722;')

    # 7. Unify borders to IV surface palette
    content = content.replace('border:1px solid #2a2a4a;', 'border:1px solid var(--border-subtle, #2a2e39);')
    content = content.replace('border:1px solid #222744;', 'border:1px solid var(--border-subtle, #2a2e39);')
    content = content.replace('border-bottom:1px solid #222744;', 'border-bottom:1px solid var(--border-subtle, #2a2e39);')
    content = content.replace('border-bottom:1px solid #2a2a4a;', 'border-bottom:1px solid var(--border-subtle, #2a2e39);')

    # 8. Clean up "⚡ NEW:" text in quick banner
    content = content.replace('<span>⚡ NEW: MARKET MAKER GAMMA EXPLOSION', '<span>MARKET MAKER GAMMA EXPLOSION')

    # 9. Remove PRO QUANT badge
    content = re.sub(
        r'\s*<span class="brand-badge">PRO QUANT</span>',
        '',
        content
    )

    # 10. Remove Institutional Gamma Explosion Quick-Launch Banner (LAUNCH TERMINAL)
    content = re.sub(
        r'\s*<!-- Institutional Gamma Explosion Quick-Launch Banner -->\s*<div class="ge-quick-banner"[\s\S]*?LAUNCH TERMINAL &rarr;\s*</div>\s*</div>',
        '',
        content
    )
    # Also handle if comment is absent or different
    content = re.sub(
        r'\s*<div class="ge-quick-banner"[\s\S]*?LAUNCH TERMINAL &rarr;\s*</div>\s*</div>',
        '',
        content
    )

    return content

def update_file(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    new_content = clean_html(content)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"Updated {filepath} ({len(content)} -> {len(new_content)} bytes)")

if __name__ == "__main__":
    for p in [
        "templates/unified_dashboard.html",
        "unified_dashboard.html",
        "unified_dashboard_fragment.html"
    ]:
        update_file(p)
