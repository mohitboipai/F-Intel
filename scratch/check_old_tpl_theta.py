import subprocess
import re
import sys

if hasattr(sys.stdout, 'reconfigure'):
    getattr(sys.stdout, 'reconfigure')(encoding='utf-8')
old_tpl = subprocess.check_output(['git', 'show', 'd626fb2:templates/unified_dashboard.html'], encoding='utf-8', errors='ignore')

m = re.search(r'(<section id="tab-theta"[^>]*>.*?</section>)', old_tpl, re.DOTALL)
if m:
    sec = m.group(1)
    print(f"Old templates tab-theta length: {len(sec)}")
    print("First 300 chars:\n", sec[:300])
    print("...")
    print("Last 300 chars:\n", sec[-300:])
else:
    print("Not found")

m_prob = re.search(r'(<section id="tab-prob"[^>]*>.*?</section>)', old_tpl, re.DOTALL)
if m_prob:
    print(f"Old templates tab-prob length: {len(m_prob.group(1))}")
