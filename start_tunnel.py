import subprocess
import socket
import re
import time
import sys
import threading
import os
import shutil
import json
from dotenv import load_dotenv

load_dotenv()

PORT = int(os.getenv("PORT", "8082"))
TUNNEL_PROVIDER = os.getenv("TUNNEL_PROVIDER", "").strip().lower()
USE_TAILSCALE = TUNNEL_PROVIDER == "tailscale" or os.getenv("USE_TAILSCALE", "").strip().lower() in ("true", "1")

TUNNEL_TOKEN = os.getenv("CLOUDFLARE_TUNNEL_TOKEN", "").strip()
TUNNEL_NAME = os.getenv("CLOUDFLARE_TUNNEL_NAME", "").strip()
HOSTNAME = os.getenv("CLOUDFLARE_HOSTNAME", "").strip() or os.getenv("CLOUDFLARE_URL", "").strip()


def get_tailscale_cmd():
    """Find tailscale executable in PATH or standard Windows install paths."""
    if shutil.which("tailscale"):
        return "tailscale"
    candidates = [
        r"C:\Program Files\Tailscale\tailscale.exe",
        r"C:\Program Files (x86)\Tailscale\tailscale.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Tailscale\tailscale.exe"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def run_tailscale_funnel():
    ts_cmd = get_tailscale_cmd()
    if not ts_cmd:
        print("Tailscale is not installed or not in PATH.")
        print("Install it with:")
        print("  winget install Tailscale.Tailscale")
        print("  or download from: https://tailscale.com/download/windows")
        sys.exit(1)

    print("Starting Tailscale Funnel on port", PORT, "...")

    # Discover the public MagicDNS name from Tailscale status
    fixed_url = os.getenv("TAILSCALE_HOSTNAME", "").strip()
    if not fixed_url:
        try:
            status_res = subprocess.run([ts_cmd, "status", "--json"], capture_output=True, text=True, timeout=5)
            if status_res.returncode == 0:
                data = json.loads(status_res.stdout)
                dns_name = data.get("Self", {}).get("DNSName", "").rstrip(".")
                if dns_name:
                    fixed_url = f"https://{dns_name}"
        except Exception:
            pass

    if not fixed_url:
        fixed_url = "https://your-machine.your-tailnet.ts.net"
    elif not fixed_url.startswith("http"):
        fixed_url = f"https://{fixed_url}"

    # Launch tailscale funnel
    proc = subprocess.Popen(
        [ts_cmd, "funnel", str(PORT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    print("\n============================================")
    print("F-Intel is ready on Tailscale Funnel\n")
    print("Open this link (accessible to everyone over HTTPS):")
    print(f"{fixed_url}\n")
    print("Permanent link: Does NOT change across restarts")
    print("============================================\n")

    for fname in (".tunnel_url", "tunnel_url.txt"):
        try:
            with open(fname, "w", encoding="utf-8") as f:
                f.write(fixed_url)
        except Exception:
            pass

    threading.Thread(target=heartbeat, daemon=True).start()

    try:
        if proc.stdout:
            for line in proc.stdout:
                # If Tailscale outputs the public URL in its logs, pass it through
                print(line, end="")
    except KeyboardInterrupt:
        pass
    finally:
        print("\nTailscale funnel stopped.")
        proc.terminate()
        sys.exit(0)


def get_cloudflared_cmd():
    """Find cloudflared in PATH or local directory."""
    if shutil.which("cloudflared"):
        return "cloudflared"
    local_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloudflared.exe")
    if os.path.exists(local_bin):
        return local_bin
    return None


def check_cloudflared():
    cmd = get_cloudflared_cmd()
    if not cmd:
        print("cloudflared is not installed or found. Install it:")
        print("Windows: download from https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/")
        print("         or place cloudflared.exe in the project folder.")
        print("Mac:     brew install cloudflared")
        print("Linux:   sudo apt install cloudflared")
        sys.exit(1)
    return cmd


def heartbeat():
    while True:
        time.sleep(300)
        print("Tunnel active")


def main():
    if USE_TAILSCALE:
        run_tailscale_funnel()
        return

    cloudflared_bin = check_cloudflared()

    # ── CASE 1: Cloudflare Named Tunnel via Token (Permanent URL) ─────────────
    if TUNNEL_TOKEN:
        print("Starting Cloudflare Named Tunnel (Permanent Token)...")
        proc = subprocess.Popen(
            [cloudflared_bin, "tunnel", "run", "--token", TUNNEL_TOKEN],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        fixed_url = HOSTNAME if HOSTNAME else "https://your-configured-domain.com"
        if not fixed_url.startswith("http"):
            fixed_url = f"https://{fixed_url}"

        print("\n============================================")
        print("F-Intel is ready on your permanent link\n")
        print(f"Open this in your phone/browser:")
        print(f"{fixed_url}\n")
        print("Permanent link: Does NOT change across restarts")
        print("============================================\n")

        for fname in (".tunnel_url", "tunnel_url.txt"):
            try:
                with open(fname, "w", encoding="utf-8") as f:
                    f.write(fixed_url)
            except Exception:
                pass

        threading.Thread(target=heartbeat, daemon=True).start()

        try:
            if proc.stdout:
                for line in proc.stdout:
                    if "INF" in line or "ERR" in line:
                        # Pass through to stdout for launcher logging
                        print(line, end="")
        except KeyboardInterrupt:
            pass
        finally:
            print("\nTunnel stopped.")
            proc.terminate()
            sys.exit(0)

    # ── CASE 2: Cloudflare Named Tunnel via CLI Tunnel Name ───────────────────
    elif TUNNEL_NAME:
        print(f"Starting Cloudflare Named Tunnel ({TUNNEL_NAME})...")
        proc = subprocess.Popen(
            [cloudflared_bin, "tunnel", "run", TUNNEL_NAME],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        fixed_url = HOSTNAME if HOSTNAME else "https://your-configured-domain.com"
        if not fixed_url.startswith("http"):
            fixed_url = f"https://{fixed_url}"

        print(f"\nPermanent link: {fixed_url}\n")
        for fname in (".tunnel_url", "tunnel_url.txt"):
            try:
                with open(fname, "w", encoding="utf-8") as f:
                    f.write(fixed_url)
            except Exception:
                pass

        threading.Thread(target=heartbeat, daemon=True).start()

        try:
            if proc.stdout:
                for line in proc.stdout:
                    print(line, end="")
        except KeyboardInterrupt:
            pass
        finally:
            print("\nTunnel stopped.")
            proc.terminate()
            sys.exit(0)

    # ── CASE 3: Quick Anonymous Tunnel (*.trycloudflare.com) ──────────────────
    else:
        print("Starting Cloudflare Quick Tunnel (ephemeral)...")
        proc = subprocess.Popen(
            [cloudflared_bin, "tunnel", "--url", f"http://localhost:{PORT}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        url_found = False

        try:
            if proc.stdout:
                for line in proc.stdout:
                    if not url_found and "https://" in line and "trycloudflare.com" in line:
                        match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
                        if match:
                            url = match.group(0)
                            print("\n============================================")
                            print("F-Intel is ready on your phone\n")
                            print(f"Open this in your phone browser:")
                            print(f"{url}\n")
                            print("Tip: bookmark it or add to home screen")
                            print("============================================\n")

                            for fname in (".tunnel_url", "tunnel_url.txt"):
                                try:
                                    with open(fname, "w", encoding="utf-8") as f:
                                        f.write(url)
                                except Exception:
                                    pass
                            url_found = True

                            threading.Thread(target=heartbeat, daemon=True).start()
        except KeyboardInterrupt:
            pass
        finally:
            print("\nTunnel stopped.")
            proc.terminate()
            sys.exit(0)


if __name__ == "__main__":
    main()

