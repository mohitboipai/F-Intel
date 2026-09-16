import re

def clean_mm_tab():
    pattern = re.compile(
        r'<!-- 1\. Interactive Real-Time Candlesticks & GEX Bands Chart.*?<!-- 3\. Strike GEX Ladder & Pin Corridor -->',
        re.DOTALL
    )
    replacement = '<!-- Strike GEX Ladder & Pin Corridor -->'

    for filepath in ['templates/unified_dashboard.html', 'unified_dashboard.html', 'unified_dashboard_fragment.html']:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if pattern.search(content):
            new_content = pattern.sub(replacement, content)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Cleaned MM tab in {filepath}")
        else:
            print(f"Pattern not found in {filepath}")

if __name__ == '__main__':
    clean_mm_tab()
