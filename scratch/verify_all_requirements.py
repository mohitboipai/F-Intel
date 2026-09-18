import sys
import urllib.request
import re

if getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
    try:
        getattr(sys.stdout, 'reconfigure', lambda **kw: None)(encoding='utf-8')
    except (AttributeError, TypeError):
        pass

def verify():
    url = "http://127.0.0.1:8082/"
    html = urllib.request.urlopen(url).read().decode('utf-8')
    
    print("=== REQUIREMENT VERIFICATION ===")
    
    # Req 1: Remove confluence verdict
    has_verdict = 'top-verdict-pill' in html or 'CONFLUENCE VERDICT' in html[:2500]
    print(f"1. Confluence Verdict removed: {not has_verdict}")
    if has_verdict:
        print("   Found verdict in header!")

    # Req 2: Remove day and night tab
    has_theme_toggle = 'theme-toggle-wrap' in html or 'theme-toggle-input' in html or 'NIGHT' in html[:2500]
    print(f"2. Day/Night mode switch removed: {not has_theme_toggle}")
    if has_theme_toggle:
        print("   Found theme toggle in header!")

    # Req 3: Design in IV Surface style
    theme_css = open('static/css/theme.css', encoding='utf-8', errors='ignore').read()
    has_cyan = bool(re.search(r'--accent:\s*#00e5ff', theme_css))
    has_slate_canvas = bool(re.search(r'--bg-canvas:\s*#131722', theme_css))
    has_slate_surface = bool(re.search(r'--bg-surface:\s*#1e222d', theme_css))
    print(f"3. IV Surface theme tokens (Cyan #00e5ff, Canvas #131722, Slate #1e222d): {has_cyan and has_slate_canvas and has_slate_surface}")

    # Req 4: Remove icon and new label from gamma tab
    tab_match = re.search(r'<button class="tab-btn mm-tab-btn" data-tab="mm">([\s\S]*?)</button>', html)
    if tab_match:
        gamma_tab_content = tab_match.group(1).strip()
        has_icon = '⚡' in gamma_tab_content
        has_new = 'NEW' in gamma_tab_content or 'pulse-badge' in gamma_tab_content
        is_clean = (not has_icon) and (not has_new) and ('GAMMA EXPLOSION & MM' in gamma_tab_content)
        print(f"4. Gamma tab clean (no icon, no NEW badge): {is_clean} (Content: {gamma_tab_content})")
    else:
        print("4. Gamma tab button NOT found!")

    # Check fragment
    frag_html = urllib.request.urlopen("http://127.0.0.1:8082/fragment").read().decode('utf-8')
    has_frag_verdict = 'frag-verdict-transfer' in frag_html or 'CONFLUENCE VERDICT' in frag_html[:2000]
    print(f"5. Fragment updates clean: {not has_frag_verdict}")

    print("================================")

if __name__ == "__main__":
    verify()
